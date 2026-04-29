import argparse
import pandas as pd
import numpy as np
from pathlib import Path
 
# ── Constantes ────────────────────────────────────────────────────────────────
EXIT_ZONES = {"Z_E1", "Z_E2", "Z_CK"}   # zonas que sinalizam fim de visita
MAX_GAP_S  = 300                          # 5 minutos: gap máximo dentro de uma visita
 
 
def load_events(path: str) -> pd.DataFrame:
    """Carrega e pré-processa o CSV de eventos."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Extrair o número do event_id para ordenação correta
    df["eid_num"] = df["event_id"].str.extract(r"e_(\d+)").astype(int)
    df = df.sort_values("eid_num").reset_index(drop=True)
    return df
 
 
def split_into_person_blocks(df: pd.DataFrame) -> list[tuple[int, int]]:
    """
    Divide o dataframe (ordenado por eid_num) em blocos, cada um
    correspondendo a uma pessoa.
 
    Critérios de corte (qualquer um é suficiente):
      - O evento anterior foi um exit numa zona de saída
      - Os atributos (gender, age_range) mudam entre eventos consecutivos
      - O gap temporal entre eventos consecutivos supera MAX_GAP_S
    """
    blocks = []
    start = 0
    n = len(df)
 
    for i in range(1, n):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
 
        # Critério 1: saída da loja no evento anterior
        exit_boundary = (
            prev["event_type"] == "exit" and prev["zone_id"] in EXIT_ZONES
        )
 
        # Critério 2: mudança de atributos demográficos
        attr_change = (
            prev["gender"] != curr["gender"]
            or prev["age_range"] != curr["age_range"]
        )
 
        # Critério 3: gap temporal excessivo
        gap_s = (curr["timestamp"] - prev["timestamp"]).total_seconds()
        timeout = gap_s > MAX_GAP_S
 
        if exit_boundary or attr_change or timeout:
            blocks.append((start, i - 1))
            start = i
 
    blocks.append((start, n - 1))
    return blocks
 
 
def build_zone_visits(person_events: pd.DataFrame) -> list[dict]:
    """
    A partir dos eventos de uma pessoa, reconstrói as visitas por zona.
 
    Para cada zona visitada, procura o triplo entry→linger→exit e extrai:
      - entry_time: timestamp do entry
      - exit_time:  timestamp do exit
      - dwell_s:    duração do linger (0 se não houver linger)
 
    Estratégia: percorre os eventos em ordem; ao encontrar um entry abre
    uma visita; ao encontrar o exit correspondente (mesma zona) fecha-a.
    """
    visits = []
    open_visits = {}  # zone_id -> {entry_time, dwell_s}
 
    for _, row in person_events.iterrows():
        zone = row["zone_id"]
        etype = row["event_type"]
 
        if etype == "entry":
            # Abre nova visita nesta zona (sobrepõe se já estava aberta — raro)
            open_visits[zone] = {
                "entry_time": row["timestamp"],
                "dwell_s": 0,
            }
 
        elif etype == "linger" and zone in open_visits:
            # Atualiza a duração de permanência
            open_visits[zone]["dwell_s"] = row["duration_s"]
 
        elif etype == "exit" and zone in open_visits:
            # Fecha a visita
            v = open_visits.pop(zone)
            visits.append({
                "zone_id":    zone,
                "entry_time": v["entry_time"],
                "exit_time":  row["timestamp"],
                "dwell_s":    v["dwell_s"],
            })
 
    # Visitas abertas sem exit correspondente (anomalia ou fim de ficheiro)
    for zone, v in open_visits.items():
        visits.append({
            "zone_id":    zone,
            "entry_time": v["entry_time"],
            "exit_time":  None,  # sem exit registado
            "dwell_s":    v["dwell_s"],
        })
 
    return visits
 
 
def reconstruct_journeys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função principal: para cada bloco de pessoa, reconstrói as visitas por
    zona e devolve um DataFrame com o schema definido no enunciado.
    """
    blocks = split_into_person_blocks(df)
    rows = []
 
    for person_idx, (start, end) in enumerate(blocks):
        person_id = f"P_{person_idx + 1:05d}"
        person_events = df.iloc[start : end + 1]
 
        # Atributos demográficos (valor mais frequente, como salvaguarda)
        gender    = person_events["gender"].mode()[0]
        age_range = person_events["age_range"].mode()[0]
 
        zone_visits = build_zone_visits(person_events)
 
        for v in zone_visits:
            entry_time = v["entry_time"]
            rows.append({
                "person_id":   person_id,
                "zone_id":     v["zone_id"],
                "entry_time":  entry_time,
                "exit_time":   v["exit_time"],
                "dwell_s":     v["dwell_s"],
                "gender":      gender,
                "age_range":   age_range,
                "visit_date":  entry_time.date() if entry_time else None,
                "hour_of_day": entry_time.hour  if entry_time else None,
            })
 
    return pd.DataFrame(rows)
 
 
def compute_quality_metrics(journeys: pd.DataFrame, n_events_original: int) -> None:
    """
    Calcula e imprime as métricas de qualidade pedidas no enunciado (Secção 3.4).
    """
    print("\n" + "=" * 55)
    print("  MÉTRICAS DE QUALIDADE DA RECONSTRUÇÃO")
    print("=" * 55)
 
    # 1. Cobertura: eventos atribuídos vs total
    # Cada visita com exit_time tem 2-3 eventos (entry, opt. linger, exit)
    # Estimativa: (visitas com exit * 2) + lingers
    n_visits_with_exit = journeys["exit_time"].notna().sum()
    n_lingers          = (journeys["dwell_s"] > 0).sum()
    n_events_covered   = n_visits_with_exit * 2 + n_lingers
    coverage = n_events_covered / n_events_original * 100
    print(f"\n[1] Cobertura:    {coverage:.1f}%  ({n_events_covered}/{n_events_original} eventos)")
 
    # 2. Consistência: nenhuma pessoa em duas zonas ao mesmo tempo
    inconsistent = 0
    for pid, grp in journeys.groupby("person_id"):
        grp_valid = grp.dropna(subset=["exit_time"]).sort_values("entry_time")
        for i in range(len(grp_valid) - 1):
            row_a = grp_valid.iloc[i]
            row_b = grp_valid.iloc[i + 1]
            # Sobreposição: entry_b < exit_a
            if row_b["entry_time"] < row_a["exit_time"]:
                inconsistent += 1
    total_pairs = len(journeys)
    consistency = (1 - inconsistent / max(total_pairs, 1)) * 100
    print(f"[2] Consistência: {consistency:.1f}%  ({inconsistent} sobreposições detetadas)")
 
    # 3. Completude: trajetórias com início em Z_E e fim em Z_E ou Z_CK
    entry_zones = {"Z_E1", "Z_E2"}
    exit_zones_all = EXIT_ZONES
 
    def check_complete(grp):
        zones = grp.sort_values("entry_time")["zone_id"]
        starts_ok = zones.iloc[0] in entry_zones if len(zones) > 0 else False
        ends_ok   = zones.iloc[-1] in exit_zones_all if len(zones) > 0 else False
        return starts_ok and ends_ok
 
    completeness_per_person = journeys.groupby("person_id").apply(check_complete)
    completeness = completeness_per_person.mean() * 100
    print(f"[3] Completude:   {completeness:.1f}%  de trajetórias com entrada e saída válidas")
 
    # 4. Plausibilidade temporal: gaps entre zonas consecutivas
    gaps = []
    for _, grp in journeys.groupby("person_id"):
        grp_sorted = grp.dropna(subset=["exit_time"]).sort_values("entry_time")
        for i in range(len(grp_sorted) - 1):
            gap = (grp_sorted.iloc[i + 1]["entry_time"] - grp_sorted.iloc[i]["exit_time"]).total_seconds()
            gaps.append(gap)
 
    if gaps:
        gaps = np.array(gaps)
        print(f"[4] Gaps entre zonas (segundos):")
        print(f"    mediana={np.median(gaps):.0f}s  média={np.mean(gaps):.0f}s  "
              f"p95={np.percentile(gaps, 95):.0f}s  max={np.max(gaps):.0f}s")
        print(f"    Gaps negativos (erro): {(gaps < 0).sum()}")
        print(f"    Gaps > 5 min:          {(gaps > 300).sum()} ({(gaps > 300).mean()*100:.1f}%)")
 
    # 5. Sumário geral
    n_persons = journeys["person_id"].nunique()
    n_zone_visits = len(journeys)
    avg_zones = n_zone_visits / n_persons if n_persons > 0 else 0
    print(f"\n[5] Sumário:")
    print(f"    Pessoas reconstruídas: {n_persons}")
    print(f"    Visitas a zonas:       {n_zone_visits}")
    print(f"    Zonas/pessoa (média):  {avg_zones:.1f}")
    print("=" * 55 + "\n")
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Reconstrói trajetórias individuais a partir do stream de eventos."
    )
    parser.add_argument("--input",  required=True, help="Caminho para events.csv")
    parser.add_argument("--output", required=True, help="Caminho para journeys.csv")
    args = parser.parse_args()
 
    print(f"[stitcher] A carregar eventos de '{args.input}'...")
    df = load_events(args.input)
    n_events = len(df)
    print(f"[stitcher] {n_events} eventos carregados.")
 
    print("[stitcher] A reconstruir trajetórias...")
    journeys = reconstruct_journeys(df)
 
    # Calcular métricas de qualidade
    compute_quality_metrics(journeys, n_events)
 
    # Guardar output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    journeys.to_csv(args.output, index=False)
    print(f"[stitcher] journeys.csv guardado em '{args.output}' ({len(journeys)} linhas).")
 
 
if __name__ == "__main__":
    main()