import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd
 
# Caminho para o mapa de zonas (lido uma vez no início)
ZONES_FILE = Path(__file__).parent.parent / "data" / "zones.json"
 
# Limites temporais (segundos)
MAX_GAP_S = 600          # Gap máximo entre zonas consecutivas (5 min)
MIN_GAP_S = 3            # Gap mínimo plausível entre zonas (evita teleporte)
MAX_IDLE_S = 1800        # Trajetória aberta sem eventos → fechar (10 min)
MAX_VISIT_DURATION_S = 5400  # Duração máxima de uma visita (90 min)

ATTR_MISMATCH_THRESHOLD = 2  # Tolerância a erros de classificação demográfica (8-12%)
 
# Score weights para o ranker de candidatos
W_GAP      = 0.5   # Proximidade temporal
W_ATTR     = 0.4   # Compatibilidade demográfica
W_ADJ      = 0.1   # Bónus por zona adjacente (trajeto plausível no mapa)
 
# Carregamento do mapa de zonas
def load_zone_graph(path: Path) -> dict:
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
 
# Tempo mínimo de deslocação entre duas zonas
def min_walk_time(graph: dict, z1: str, z2: str) -> int:
    # Se as zonas forem as mesmas
    if z1 == z2:
        return 0
    # Se as zonas são adjacentes no mapa
    if z1 in graph and z2 in graph[z1]:
        return graph[z1][z2]
    # Se as zonas não são adjacentes no mapa, então encontra um caminho com zona intermédia
    if z1 in graph:
        for mid, t1 in graph[z1].items():
            if mid in graph and z2 in graph[mid]:
                return t1 + graph[mid][z2]
    # Se não encontrar nada
    return MIN_GAP_S
 
# Lista de adjacências
def is_adjacent(graph: dict, z1: str, z2: str) -> bool:
    return z1 in graph and z2 in graph[z1]
 
# Estrutura dos dados
@dataclass
class OpenTrajectory:
    person_id: str                  # Exemplo: "P_00001"
    gender: str                     # "M" ou "F"
    age_range: str                  # Exemplo: "adult"
 
    last_zone: str                  # Zona atual
    start_ts: pd.Timestamp          # Timestamp de início da visita à loja
    last_entry_ts: pd.Timestamp     # Quando entrou na zona atual
    last_exit_ts: Optional[pd.Timestamp]   # Quando saiu (None se ainda dentro da zona)
 
    visits: list = field(default_factory=list) # Lista de todas as zonas visitadas
 
    attr_mismatches: int = 0        # Contador de inconsistências demográficas detetadas
 
    closed: bool = False # True se já saiu da loja
    
    # Timestamp mais recente da trajetória
    def current_ts(self) -> pd.Timestamp:
        # Se já saiu da zona, então usa exit_time
        if self.last_exit_ts is not None:
            return self.last_exit_ts
        # Se ainda está na zona, então usa entry_time
        return self.last_entry_ts
    
    # Se a pessoa ainda não saiu da última zona detetada
    def is_in_zone(self) -> bool:
        return self.last_exit_ts is None
    
    # Compatibilidade demográfica entre a trajetória e um novo evento
    def attr_score(self, gender: str, age_range: str) -> float:
        # 1.0 = género e idade coincidem
        # 0.5 = só um coincide
        # 0.0 = dois diferentes
        g_ok = (self.gender == gender)
        a_ok = (self.age_range == age_range)
        return (g_ok + a_ok) / 2.0
 
# Stitcher
class Stitcher:
    def __init__(self, zone_graph: dict):
        self.graph = zone_graph
        # Substitui open_trajs por índice zona → trajetórias
        self.open_trajs_by_zone: dict[str, list[OpenTrajectory]] = defaultdict(list)
        self.closed_trajs: list[OpenTrajectory] = []
        self._person_counter = 0
        self._unmatched_entry = 0
        self._unmatched_exit = 0
        self._unmatched_linger = 0

    # Helper para iterar sobre todas as trajetórias abertas
    def _all_open(self) -> list[OpenTrajectory]:
        return [t for trajs in self.open_trajs_by_zone.values() for t in trajs]
    
    # Gera um ID sintético para cada pessoa
    def _new_person_id(self) -> str:
        self._person_counter += 1
        return f"P_{self._person_counter:05d}"
    
    # Decide se um evento pode pertencer a uma trajetória aberta e um novo evento entry
    def _score_candidate(
        self,
        traj: OpenTrajectory,
        event_ts: pd.Timestamp,
        zone_id: str,
        gender: str,
        age_range: str,
    ) -> float:
        
        # Score mais alto = melhor candidato.
        # Retorna -inf se a transição violar um hard constraint.
        gap = (event_ts - traj.current_ts()).total_seconds()
 
        """ Fase 1: eliminação imediata"""
        # Filtro 1: sobreposição temporal - pessoa ainda está dentro de uma zona
        if traj.is_in_zone() and gap < 5:
            return float("-inf")
 
        # Filtro 2: gap demasiado grande - trajetória provavelmente já terminou
        if gap > MAX_GAP_S:
            return float("-inf")
 
        # Filtro 3: chegou rápido demais - verificar se é fisicamente possível chegar à zona
        walk = min_walk_time(self.graph, traj.last_zone, zone_id)
        if gap < walk * 0.6:  # margem de 40% para correr/sensor lag
            return float("-inf")
 
        # Filtro 4: demasiados erros demográficos - associação de eventos de pessoas diferentes
        if traj.attr_mismatches >= ATTR_MISMATCH_THRESHOLD:
            attr_s = traj.attr_score(gender, age_range)
            if attr_s < 0.5:  # ambos os atributos divergem
                return float("-inf")

        """ Fase 2: decidir qual é o melhor candidato"""
        # Penalidade pelo gap (normalizado entre 0 e MAX_GAP_S)
        gap_score = 1.0 - (gap / MAX_GAP_S)
        # Bónus de atributos
        attr_s = traj.attr_score(gender, age_range)
        # Bónus de adjacência
        adj_bonus = 1.0 if is_adjacent(self.graph, traj.last_zone, zone_id) else 0.0
 
        return W_GAP * gap_score + W_ATTR * attr_s + W_ADJ * adj_bonus

    # Mantém lista de trajetórias abertas pequena         
    # Fecha trajetórias abertas há mais de MAX_IDLE_S sem atividade.
    def _expire_old_trajs(self, current_ts: pd.Timestamp):
        new_index = defaultdict(list)
        for zone, trajs in self.open_trajs_by_zone.items():
            for traj in trajs:
                idle = (current_ts - traj.current_ts()).total_seconds()
                total = (current_ts - traj.start_ts).total_seconds()
                # Se a trajetória não recebe eventos há mais de 10 minutos e dura mais de 90 minutos desde o começo, fechar
                if idle > MAX_IDLE_S or total > MAX_VISIT_DURATION_S:
                    traj.closed = True
                    self.closed_trajs.append(traj)
                else:
                    new_index[zone].append(traj)
        self.open_trajs_by_zone = new_index
 
    # Processa um evento entry e associa-o à melhor trajetória aberta
    def process_entry(self, row: pd.Series):
        ts        = row["timestamp"]
        zone_id   = row["zone_id"]
        gender    = row["gender"]
        age_range = row["age_range"]
        event_id  = row["event_id"]
 
        self._expire_old_trajs(ts) # Limpa trajetórias antigas
 
        # Se a trajetória começar em Z_E, cria uma nova trajetória
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
            self.open_trajs_by_zone[zone_id].append(traj)
            return
 
        # Se começar noutra, percorrer todas as trajetórias abertas
        best_traj  = None
        best_score = float("-inf")

        # Calcular score de cada trajetória aberta e associar ao melhor candidato
        for traj in self._all_open():
            score = self._score_candidate(traj, ts, zone_id, gender, age_range)
            if score > best_score:
                best_score = score
                best_traj  = traj
        
        # Se nenhuma trajetória for compatível, criar trajetória incompleta
        if best_traj is None:
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
                "zone_id":    zone_id,
                "entry_time": ts,
                "exit_time":  None,
                "dwell_s":    0,
                "event_ids":  [event_id],
            })
            self.open_trajs_by_zone[zone_id].append(traj)
            return
 
        # Registar inconsistência demográfica se existir
        if best_traj.attr_score(gender, age_range) < 1.0:
            best_traj.attr_mismatches += 1
 
        # Remove da zona antiga, adiciona na nova
        self.open_trajs_by_zone[best_traj.last_zone].remove(best_traj)
        best_traj.last_zone      = zone_id
        best_traj.last_entry_ts  = ts
        best_traj.last_exit_ts   = None
        best_traj.visits.append({
            "zone_id":    zone_id,
            "entry_time": ts,
            "exit_time":  None,
            "dwell_s":    0,
            "event_ids":  [event_id],
        })
        self.open_trajs_by_zone[zone_id].append(best_traj)

# Processa um evento exit, atualizando a última visita da trajetória correspondente na mesma zona.
    def process_exit(self, row: pd.Series):
        ts        = row["timestamp"]
        zone_id   = row["zone_id"]
        gender    = row["gender"]
        age_range = row["age_range"] # Adicionado para melhor matching
        event_id  = row["event_id"]  # Adicionado para guardar o evento

        candidates = self.open_trajs_by_zone.get(zone_id, [])
        best, best_score = None, (float("inf"), float("inf"))

        # 1. TENTATIVA NORMAL: Procurar trajetória aberta na MESMA ZONA
        for traj in candidates:
            # Se ainda não tem saída registada
            if traj.last_exit_ts is not None:
                continue
            # O entry_time está mais próximo do exit_time atual
            gap = abs((ts - traj.last_entry_ts).total_seconds())
            # Desempate por género
            gender_bonus = 0 if traj.gender == gender else 1
            score        = (gap, gender_bonus)
            if score < best_score:
                best_score = score
                best       = traj
 
        # 2. SALVAGUARDA (Inferred Entry): Câmaras falharam a entrada!
        if best is None:
            best_inferred = None
            best_inferred_score = float("-inf")
            
            # Procurar em TODAS as pessoas ativas na loja, independentemente da zona
            for traj in self._all_open():
                # Ignorar se já estiver avaliado (na mesma zona mas com saída registada, etc)
                if traj.last_zone == zone_id and traj.last_exit_ts is None:
                    continue
                
                gap = (ts - traj.current_ts()).total_seconds()
                walk = min_walk_time(self.graph, traj.last_zone, zone_id)
                
                # Regras de plausibilidade:
                # O gap temporal tem de dar tempo para chegar lá (com 50% de margem) e não ser gigante
                if gap > (walk * 0.5) and gap < MAX_GAP_S:
                    # Avaliar compatibilidade de idade e género
                    attr_score = traj.attr_score(gender, age_range)
                    if attr_score >= 0.5: # Tem de ter pelo menos 1 atributo igual
                        # Preferir quem estava demograficamente mais próximo
                        if attr_score > best_inferred_score:
                            best_inferred_score = attr_score
                            best_inferred = traj
            
            if best_inferred is not None:
                # Encontrámos um bom candidato noutra zona.
                # Vamos forçar a entrada ("inferir" que ele entrou instantes antes de sair)
                self.open_trajs_by_zone[best_inferred.last_zone].remove(best_inferred)
                best_inferred.last_zone = zone_id
                best_inferred.last_entry_ts = ts - pd.Timedelta(seconds=1) # Finge que entrou 1s antes
                best_inferred.visits.append({
                    "zone_id": zone_id,
                    "entry_time": best_inferred.last_entry_ts,
                    "exit_time": None,
                    "dwell_s": 0,
                    "event_ids": [] # A lista começa vazia porque falhámos o event_id do entry
                })
                self.open_trajs_by_zone[zone_id].append(best_inferred)
                best = best_inferred
            else:
                # Se não encontrarmos mesmo ninguém compatível, aí sim, contamos como órfão
                self._unmatched_exit += 1
                return
 
        # 3. Atualiza a última visita com exit_time e calcula tempo de permanência (dwell_s)
        best.last_exit_ts = ts
        if best.visits:
            last_visit = best.visits[-1]
            if last_visit["zone_id"] == zone_id and last_visit["exit_time"] is None:
                last_visit["exit_time"] = ts
                entry = last_visit["entry_time"]
                last_visit["dwell_s"] = int((ts - entry).total_seconds())
                last_visit["event_ids"].append(event_id) # Guardamos o exit
 
        # 4. Fechar trajetória se a saída for por porta ou caixa
        if zone_id in ("Z_E1", "Z_E2", "Z_CK"):
            self.open_trajs_by_zone[zone_id].remove(best)
            best.closed = True
            self.closed_trajs.append(best)

# Processa um evento do tipo linger (pessoa parada numa zona)
    def process_linger(self, row: pd.Series):
        ts        = row["timestamp"]
        zone_id   = row["zone_id"]
        duration  = row["duration_s"]
        gender    = row["gender"]
        age_range = row["age_range"] # Adicionado para melhor matching
        event_id  = row["event_id"]

        candidates = self.open_trajs_by_zone.get(zone_id, [])
        best, best_score = None, (float("inf"), float("inf"))

        # 1. TENTATIVA NORMAL: Procurar na mesma zona
        for traj in candidates:
            if traj.last_exit_ts is not None:
                continue
            gap          = abs((ts - traj.last_entry_ts).total_seconds())
            gender_bonus = 0 if traj.gender == gender else 1
            score        = (gap, gender_bonus)
            if score < best_score:
                best_score = score
                best       = traj

        # 2. SALVAGUARDA (Inferred Entry): O sensor falhou a entrada, mas a pessoa está aqui parada!
        if best is None:
            best_inferred = None
            best_inferred_score = float("-inf")
            
            for traj in self._all_open():
                if traj.last_zone == zone_id and traj.last_exit_ts is None:
                    continue
                    
                gap = (ts - traj.current_ts()).total_seconds()
                walk = min_walk_time(self.graph, traj.last_zone, zone_id)
                
                # Regras de plausibilidade
                if gap > (walk * 0.5) and gap < MAX_GAP_S:
                    attr_score = traj.attr_score(gender, age_range)
                    if attr_score >= 0.5:
                        if attr_score > best_inferred_score:
                            best_inferred_score = attr_score
                            best_inferred = traj
                            
            if best_inferred is not None:
                # Pescamos o candidato e forçamos a entrada
                self.open_trajs_by_zone[best_inferred.last_zone].remove(best_inferred)
                best_inferred.last_zone = zone_id
                best_inferred.last_entry_ts = ts - pd.Timedelta(seconds=1) # Entrou instantes antes
                best_inferred.visits.append({
                    "zone_id": zone_id,
                    "entry_time": best_inferred.last_entry_ts,
                    "exit_time": None,
                    "dwell_s": 0,
                    "event_ids": [] 
                })
                self.open_trajs_by_zone[zone_id].append(best_inferred)
                best = best_inferred
            else:
                # Ninguém compatível, é um fantasma
                self._unmatched_linger += 1
                return

        # 3. ATUALIZAR TEMPO DE PERMANÊNCIA (dwell_s)
        if best.visits:
            last = best.visits[-1]
            if last["zone_id"] == zone_id:
                # Atualiza a duração
                last["dwell_s"] = max(last["dwell_s"], duration)
                # Guarda o event_id deste linger
                if event_id not in last["event_ids"]:
                    last["event_ids"].append(event_id)
 
    # Fecha todas as trajetórias ainda abertas no final do dataset
    def flush(self):
        for traj in self._all_open():
            traj.closed = True
            self.closed_trajs.append(traj)
        self.open_trajs_by_zone.clear()
    
    # Devolve lista completa de trajetórias fechadas para ser convertida em csv
    def all_trajectories(self) -> list[OpenTrajectory]:
        return self.closed_trajs
 
# Construção do output journeys.csv
def build_journeys_df(trajs: list[OpenTrajectory]) -> pd.DataFrame:
    # Lista com o esquema pretendido
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
 
# Cálculo das métricas de qualidade
def compute_quality_metrics(df: pd.DataFrame, total_events: int, unmatched: int) -> dict:
    n_trajs = df["person_id"].nunique()
 
    # Cobertura: eventos atribuídos / total de eventos no csv
    coverage = 1.0 - (unmatched / total_events) if total_events > 0 else 0.0
 
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
 
    # Completude: trajetórias com início em Z_E e fim em Z_E ou Z_CK
    entrance_zones = {"Z_E1", "Z_E2"}
    exit_zones     = {"Z_E1", "Z_E2", "Z_CK"}
    first_zones = df.groupby("person_id")["zone_id"].first()
    last_zones  = df.groupby("person_id")["zone_id"].last()
    starts_ok = first_zones.isin(entrance_zones).sum()
    ends_ok   = last_zones.isin(exit_zones).sum()
    completeness = (starts_ok + ends_ok) / (2 * n_trajs) if n_trajs > 0 else 0
 
    # Gap distribution: mediana e percentil 95 dos gaps entre zonas consecutivas
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
    total_unmatched = stitcher._unmatched_exit + stitcher._unmatched_linger
    metrics = compute_quality_metrics(journeys, total_events, total_unmatched)
    percent_keys = {"coverage", "consistency", "completeness"}
    print("\n Métricas de qualidade ")
    for k, v in metrics.items():
        if k in percent_keys and v is not None:
            print(f"  {k:<30} {v * 100:.1f}%")
        else:
            print(f"  {k:<30} {v}")

    # Distribuição de eventos não associados
    print("\n Eventos não associados ")
    print(f"  {'exit':<30} {stitcher._unmatched_exit:,}")
    print(f"  {'linger':<30} {stitcher._unmatched_linger:,}")
    print(f"  {'total':<30} {total_unmatched:,} ({total_unmatched/total_events*100:.1f}%)")
    print(f"\nTempo total: {time.time() - t0:.1f}s")
 
 
if __name__ == "__main__":
    main()