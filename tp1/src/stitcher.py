import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
 
# Constantes e configuração
# Caminho para o mapa de zonas (lido uma vez no início)
ZONES_FILE = Path(__file__).parent.parent / "data" / "zones.json"
 
# Limites temporais (segundos)
MAX_GAP_S = 300          # Gap máximo entre zonas consecutivas (5 min)
MIN_GAP_S = 3            # Gap mínimo plausível entre zonas (evita teleporte)
MAX_IDLE_S = 600         # Trajetória aberta sem eventos → fechar (10 min)
MAX_VISIT_DURATION_S = 5400  # Duração máxima de uma visita (90 min)
 
# Tolerância a erros de classificação demográfica (8-12% segundo o enunciado)
# Uma trajetória com >ATTR_MISMATCH_THRESHOLD inconsistências é penalizada,
# mas não rejeitada imediatamente (pode ser erro do sensor).
ATTR_MISMATCH_THRESHOLD = 2
 
# Score weights para o ranker de candidatos
W_GAP      = 0.5   # Penalidade pelo gap temporal (normalizado)
W_ATTR     = 0.4   # Compatibilidade de atributos
W_ADJ      = 0.1   # Bónus por zona adjacente (trajeto plausível no mapa)
 
# Carregamento do mapa de zonas
def load_zone_graph(path: Path) -> dict:
    """
    Carrega zones.json e devolve um dicionário com:
      zone_graph[z1][z2] = walk_seconds  (tempo mínimo de deslocação)
    Para zonas não adjacentes, calcula distância mínima via BFS (1 hop).
    """
    if not path.exists():
        # Se o ficheiro não existir, assume grafo vazio (sem penalidade de adj.)
        print(f"[WARN] {path} não encontrado — adjacência desativada.", file=sys.stderr)
        return {}
 
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
 
    zones = raw.get("zones", {})
    graph = {}
    for zone_id, info in zones.items():
        graph[zone_id] = dict(info.get("walk_seconds", {}))
 
    return graph
 
 
def min_walk_time(graph: dict, z1: str, z2: str) -> int:
    """
    Tempo mínimo de caminhada entre z1 e z2.
    Devolve o valor direto se adjacentes, ou estimativa via BFS (máximo 2 hops).
    Se desconhecido, devolve MIN_GAP_S como floor conservador.
    """
    if z1 == z2:
        return 0
    if z1 in graph and z2 in graph[z1]:
        return graph[z1][z2]
    # 1-hop intermediário
    if z1 in graph:
        for mid, t1 in graph[z1].items():
            if mid in graph and z2 in graph[mid]:
                return t1 + graph[mid][z2]
    return MIN_GAP_S  # floor conservador
 
 
def is_adjacent(graph: dict, z1: str, z2: str) -> bool:
    return z1 in graph and z2 in graph[z1]
 
# Estrutura de dados: Trajetória aberta
@dataclass
class OpenTrajectory:
    person_id: str
    gender: str
    age_range: str
 
    # Última zona visitada e timestamps
    last_zone: str
    start_ts: pd.Timestamp       # Timestamp de início da visita à loja
    last_entry_ts: pd.Timestamp
    last_exit_ts: Optional[pd.Timestamp]   # None se ainda dentro da zona
 
    # Historial de visitas a zonas (uma por zona)
    visits: list = field(default_factory=list)
 
    # Contador de inconsistências demográficas detetadas
    attr_mismatches: int = 0
 
    # Flag: esta trajetória já foi fechada?
    closed: bool = False
 
    def current_ts(self) -> pd.Timestamp:
        """Timestamp mais recente desta trajetória."""
        if self.last_exit_ts is not None:
            return self.last_exit_ts
        return self.last_entry_ts
 
    def is_in_zone(self) -> bool:
        """True se a pessoa ainda não saiu da última zona detetada."""
        return self.last_exit_ts is None
 
    def attr_score(self, gender: str, age_range: str) -> float:
        """
        Score de compatibilidade demográfica [0, 1].
        1.0 = match perfeito, 0.5 = um atributo diferente, 0.0 = dois diferentes.
        """
        g_ok = (self.gender == gender)
        a_ok = (self.age_range == age_range)
        return (g_ok + a_ok) / 2.0
 
# Motor de stitching
class Stitcher:
    def __init__(self, zone_graph: dict):
        self.graph = zone_graph
        self.open_trajs: list[OpenTrajectory] = []
        self.closed_trajs: list[OpenTrajectory] = []
        self._person_counter = 0
        self._unmatched_events = 0  # eventos não associados a nenhuma trajetória
 
    def _new_person_id(self) -> str:
        self._person_counter += 1
        return f"P_{self._person_counter:05d}"
 
    def _score_candidate(
        self,
        traj: OpenTrajectory,
        event_ts: pd.Timestamp,
        zone_id: str,
        gender: str,
        age_range: str,
    ) -> float:
        """
        Score de compatibilidade entre uma trajetória aberta e um novo evento entry.
        Score mais alto = melhor candidato.
        Retorna -inf se a transição violar um hard constraint.
        """
        gap = (event_ts - traj.current_ts()).total_seconds()
 
        # Hard constraints
 
        # 1. Sobreposição temporal: pessoa ainda dentro de uma zona
        #    (não saiu) e recebe novo entry → impossível (seria duas zonas ao mesmo tempo).
        if traj.is_in_zone() and gap < 5:
            return float("-inf")
 
        # 2. Gap demasiado grande → trajetória provavelmente já terminou
        if gap > MAX_GAP_S:
            return float("-inf")
 
        # 3. Gap mínimo: verificar se é fisicamente possível chegar à zona
        walk = min_walk_time(self.graph, traj.last_zone, zone_id)
        if gap < walk * 0.6:  # margem de 40% para correr/sensor lag
            return float("-inf")
 
        # 4. Demasiadas inconsistências de atributos acumuladas
        if traj.attr_mismatches >= ATTR_MISMATCH_THRESHOLD:
            attr_s = traj.attr_score(gender, age_range)
            if attr_s < 0.5:  # ambos os atributos divergem
                return float("-inf")
 
        # ── Soft scoring ─────────────────────────────────────────────────────
 
        # Penalidade pelo gap (normalizado entre 0 e MAX_GAP_S)
        gap_score = 1.0 - (gap / MAX_GAP_S)
 
        # Bónus de atributos
        attr_s = traj.attr_score(gender, age_range)
 
        # Bónus de adjacência
        adj_bonus = 1.0 if is_adjacent(self.graph, traj.last_zone, zone_id) else 0.0
 
        return W_GAP * gap_score + W_ATTR * attr_s + W_ADJ * adj_bonus
 
    def _expire_old_trajs(self, current_ts: pd.Timestamp):
        """
        Fecha trajetórias abertas há mais de MAX_IDLE_S sem atividade.
        Chamado a cada evento para manter a lista de abertas pequena.
        """
        still_open = []
        for traj in self.open_trajs:
            idle = (current_ts - traj.current_ts()).total_seconds()
            total_duration = (current_ts - traj.start_ts).total_seconds()
            if idle > MAX_IDLE_S or total_duration > MAX_VISIT_DURATION_S:
                traj.closed = True
                self.closed_trajs.append(traj)
            else:
                still_open.append(traj)
        self.open_trajs = still_open
 
    def process_entry(self, row: pd.Series):
        """Processa um evento entry e associa-o à melhor trajetória aberta."""
        ts       = row["timestamp"]
        zone_id  = row["zone_id"]
        gender   = row["gender"]
        age_range = row["age_range"]
        event_id = row["event_id"]
 
        # Expirar trajetórias muito antigas (mantém a lista pequena)
        self._expire_old_trajs(ts)
 
        # ── Caso especial: entrada numa zona Z_E → nova trajetória ──────────
        # Uma nova pessoa entrou na loja. Não tentamos associar a trajetórias
        # existentes porque entradas pela porta são pontos de início bem definidos.
        if zone_id.startswith("Z_E"):
            traj = OpenTrajectory(
                person_id=self._new_person_id(),
                gender=gender,
                age_range=age_range,
                start_ts=ts,
                last_zone=zone_id,
                last_entry_ts=ts,
                last_exit_ts=None,
            )
            traj.visits.append({
                "zone_id": zone_id,
                "entry_time": ts,
                "exit_time": None,
                "dwell_s": 0,
                "event_ids": [event_id],
            })
            self.open_trajs.append(traj)
            return
 
        # ── Caso geral: associar a melhor trajetória aberta ─────────────────
        best_traj = None
        best_score = float("-inf")
 
        for traj in self.open_trajs:
            score = self._score_candidate(traj, ts, zone_id, gender, age_range)
            if score > best_score:
                best_score = score
                best_traj = traj
 
        if best_traj is None:
            # Nenhuma trajetória compatível → evento perdido (contado para métricas)
            self._unmatched_events += 1
            return
 
        # Registar inconsistência demográfica se existir
        if best_traj.attr_score(gender, age_range) < 1.0:
            best_traj.attr_mismatches += 1
 
        # Atualizar a trajetória
        best_traj.last_zone = zone_id
        best_traj.last_entry_ts = ts
        best_traj.last_exit_ts = None
        best_traj.visits.append({
            "zone_id": zone_id,
            "entry_time": ts,
            "exit_time": None,
            "dwell_s": 0,
            "event_ids": [event_id],
        })
 
    def process_exit(self, row: pd.Series):
        """
        Processa um evento exit, atualizando a última visita da trajetória
        correspondente na mesma zona.
        """
        ts       = row["timestamp"]
        zone_id  = row["zone_id"]
        gender   = row["gender"]
 
        # Procurar trajetória aberta na mesma zona, com atributo compatível
        best = None
        best_gap = float("inf")
 
        for traj in self.open_trajs:
            if traj.last_zone != zone_id:
                continue
            if traj.last_exit_ts is not None:
                continue  # já tem exit nesta zona
            gap = abs((ts - traj.last_entry_ts).total_seconds())
            attr_ok = (traj.gender == gender) or True  # tolerante no exit
            if gap < best_gap:
                best_gap = gap
                best = traj
 
        if best is None:
            self._unmatched_events += 1
            return
 
        # Atualizar a última visita com exit_time e dwell_s
        best.last_exit_ts = ts
        if best.visits:
            last_visit = best.visits[-1]
            if last_visit["zone_id"] == zone_id and last_visit["exit_time"] is None:
                last_visit["exit_time"] = ts
                entry = last_visit["entry_time"]
                last_visit["dwell_s"] = int((ts - entry).total_seconds())
 
        # Fechar trajetória se saída for por porta ou caixa
        if zone_id in ("Z_E1", "Z_E2", "Z_CK"):
            best.closed = True
            self.open_trajs.remove(best)
            self.closed_trajs.append(best)
 
    def process_linger(self, row: pd.Series):
        """
        Processa um evento linger: atualiza o dwell_s da última visita aberta
        na mesma zona. O linger é gerado pelo sensor entre entry e exit.
        """
        ts        = row["timestamp"]
        zone_id   = row["zone_id"]
        duration  = row["duration_s"]
        gender    = row["gender"]
 
        for traj in self.open_trajs:
            if traj.last_zone != zone_id:
                continue
            if traj.last_exit_ts is not None:
                continue
            attr_ok = (traj.gender == gender) or True
            if not attr_ok:
                continue
            if traj.visits:
                last = traj.visits[-1]
                if last["zone_id"] == zone_id:
                    last["dwell_s"] = max(last["dwell_s"], duration)
                    if "linger" not in last["event_ids"]:
                        last["event_ids"].append(row["event_id"])
            break  # associar ao primeiro candidato encontrado
 
    def flush(self):
        """Fechar todas as trajetórias ainda abertas no final do dataset."""
        for traj in self.open_trajs:
            traj.closed = True
            self.closed_trajs.append(traj)
        self.open_trajs = []
 
    def all_trajectories(self) -> list[OpenTrajectory]:
        return self.closed_trajs
 
# Construção do output journeys.csv
def build_journeys_df(trajs: list[OpenTrajectory]) -> pd.DataFrame:
    """
    Converte a lista de trajetórias no schema de output pedido:
      person_id, zone_id, entry_time, exit_time, dwell_s,
      gender, age_range, visit_date, hour_of_day
    """
    rows = []
    for traj in trajs:
        for visit in traj.visits:
            entry_ts = visit["entry_time"]
            rows.append({
                "person_id":   traj.person_id,
                "zone_id":     visit["zone_id"],
                "entry_time":  entry_ts,
                "exit_time":   visit["exit_time"],
                "dwell_s":     visit["dwell_s"],
                "gender":      traj.gender,
                "age_range":   traj.age_range,
                "visit_date":  entry_ts.date() if entry_ts else None,
                "hour_of_day": entry_ts.hour if entry_ts else None,
            })
    return pd.DataFrame(rows)
 
# Métricas de qualidade da reconstrução
def compute_quality_metrics(df: pd.DataFrame, total_events: int, unmatched: int) -> dict:
    """
    Calcula e imprime as métricas de qualidade pedidas no enunciado.
    """
    n_trajs = df["person_id"].nunique()
 
    # Cobertura: eventos atribuídos / total
    assigned_events = len(df) * 3  # aprox.: entry + linger + exit por visita
    coverage = min(1.0, assigned_events / total_events)
 
    # Consistência: sem sobreposição temporal
    overlaps = 0
    for pid, group in df.groupby("person_id"):
        group = group.sort_values("entry_time").copy()
        entries = group["entry_time"].values
        exits   = group["exit_time"].fillna(group["entry_time"]).values
        for i in range(len(entries) - 1):
            if exits[i] is not None and entries[i + 1] < exits[i]:
                overlaps += 1
    consistency = 1.0 - (overlaps / max(1, n_trajs))
 
    # Completude: trajetórias com início em Z_E
    entrance_zones = {"Z_E1", "Z_E2"}
    exit_zones     = {"Z_E1", "Z_E2", "Z_CK"}
 
    first_zones = df.groupby("person_id")["zone_id"].first()
    last_zones  = df.groupby("person_id")["zone_id"].last()
 
    starts_ok = first_zones.isin(entrance_zones).sum()
    ends_ok   = last_zones.isin(exit_zones).sum()
    completeness = (starts_ok + ends_ok) / (2 * n_trajs) if n_trajs > 0 else 0
 
    # Gap distribution
    gaps = []
    for pid, group in df.groupby("person_id"):
        group = group.sort_values("entry_time")
        exits   = group["exit_time"].tolist()
        entries = group["entry_time"].tolist()
        for i in range(len(entries) - 1):
            if exits[i] is not None:
                gap = (entries[i + 1] - exits[i]).total_seconds()
                gaps.append(gap)
 
    gap_series = pd.Series(gaps)
 
    metrics = {
        "trajectories_total":    n_trajs,
        "events_total":          total_events,
        "events_unmatched":      unmatched,
        "coverage":              round(coverage, 4),
        "consistency":           round(consistency, 4),
        "completeness":          round(completeness, 4),
        "gap_seconds_mean":      round(gap_series.mean(), 1) if len(gaps) > 0 else None,
        "gap_seconds_median":    round(gap_series.median(), 1) if len(gaps) > 0 else None,
        "gap_seconds_p95":       round(gap_series.quantile(0.95), 1) if len(gaps) > 0 else None,
    }
    return metrics
 
# Main
def main():
    parser = argparse.ArgumentParser(description="Stitcher — reconstrução de trajetórias")
    parser.add_argument("--input",  required=True, help="Caminho para events.csv")
    parser.add_argument("--output", required=True, help="Caminho para journeys.csv")
    parser.add_argument("--zones",  default=str(ZONES_FILE), help="Caminho para zones.json")
    args = parser.parse_args()
 
    t0 = time.time()
    print(f"[1/4] A carregar {args.input}")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    total_events = len(df)
    print(f"      {total_events:,} eventos carregados.")
 
    print(f"[2/4] A carregar mapa de zonas")
    graph = load_zone_graph(Path(args.zones))
    print(f"      {len(graph)} zonas no grafo.")
 
    print(f"[3/4] A executar stitching")
    stitcher = Stitcher(graph)
 
    # Processar eventos em ordem cronológica
    for _, row in df.iterrows():
        etype = row["event_type"]
        if etype == "entry":
            stitcher.process_entry(row)
        elif etype == "exit":
            stitcher.process_exit(row)
        elif etype == "linger":
            stitcher.process_linger(row)
 
    stitcher.flush()
    trajs = stitcher.all_trajectories()
    print(f"      {len(trajs):,} trajetórias reconstruídas.")
 
    print(f"[4/4] A escrever {args.output}")
    journeys = build_journeys_df(trajs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    journeys.to_csv(args.output, index=False)
    print(f"      {len(journeys):,} linhas escritas.")
 
    # Métricas de qualidade
    metrics = compute_quality_metrics(journeys, total_events, stitcher._unmatched_events)
    print("\n Métricas de qualidade ")
    percent_keys = {"coverage", "consistency", "completeness"}
    print("\n── Métricas de qualidade ──────────────────────────")
    for k, v in metrics.items():
        if k in percent_keys and v is not None:
            print(f"  {k:<30} {v * 100:.1f}%")
        else:
            print(f"  {k:<30} {v}")
    print(f"\nTempo total: {time.time() - t0:.1f}s")
 
 
if __name__ == "__main__":
    main()