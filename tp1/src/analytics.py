import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
import pandas as pd
import numpy as np
 
warnings.filterwarnings("ignore")
 
# Helpers
CHECKOUT_ZONES  = {"Z_C1", "Z_C2", "Z_C3", "Z_CK"}
ENTRANCE_ZONES  = {"Z_E1", "Z_E2"}
SECTION_ZONES   = {f"Z_S{i}" for i in range(1, 8)}
NAV_ZONES       = {f"Z_N{i}" for i in range(1, 11)}
 
DAY_NAMES = {
    0: "segunda", 1: "terça", 2: "quarta", 3: "quinta",
    4: "sexta",   5: "sábado", 6: "domingo"
}
 
ZONE_DESCRIPTIONS = {
    "Z_E1": "Entrada principal Norte — lado esquerdo",
    "Z_E2": "Entrada principal Norte — lado direito",
    "Z_C1": "Caixas registadoras — bloco esquerdo",
    "Z_C2": "Caixas registadoras — bloco central",
    "Z_C3": "Caixas registadoras — bloco direito",
    "Z_CK": "Corredor de saída após pagamento",
    "Z_N1": "Corredor — ala esquerda, linha 1",
    "Z_N2": "Corredor — centro, linha 1",
    "Z_N3": "Corredor — ala direita, linha 1",
    "Z_N4": "Corredor — ala esquerda, linha 2",
    "Z_N5": "Corredor — centro, linha 2",
    "Z_N6": "Corredor — ala direita, linha 2",
    "Z_N7": "Corredor — ala esquerda, fundo",
    "Z_N8": "Corredor — centro, fundo",
    "Z_N9": "Corredor — ala direita, fundo",
    "Z_N10": "Corredor traseiro — parede Sul",
    "Z_S1": "Secção de frescos e lacticínios",
    "Z_S2": "Secção de padaria e pastelaria",
    "Z_S3": "Secção de talho e charcutaria",
    "Z_S4": "Secção de produtos de higiene e limpeza",
    "Z_S5": "Secção de bebidas e conservas",
    "Z_S6": "Secção de vinhos e destilados",
    "Z_S7": "Secção de produtos congelados",
}
 
 
def safe_round(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), decimals)
 
 
def load_journeys(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["day_of_week"] = df["visit_date"].dt.dayofweek
    df["day_name"]    = df["day_of_week"].map(DAY_NAMES)
    df["day_num"]     = (df["visit_date"] - df["visit_date"].min()).dt.days + 1
    return df
 
# 1. Métricas de tráfego
def compute_traffic(df: pd.DataFrame) -> dict:
    # Visitantes únicos por dia
    by_day = (
        df.groupby(["visit_date", "day_name"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "visitors"})
    )
    visitors_by_day = [
        {
            "date": str(row.visit_date.date()),
            "day_name": row.day_name,
            "visitors": int(row.visitors),
        }
        for row in by_day.itertuples()
    ]
 
    # Visitantes por hora (agregado toda a semana)
    by_hour = (
        df.groupby("hour_of_day")["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "visitors"})
    )
    visitors_by_hour = [
        {"hour": int(r.hour_of_day), "visitors": int(r.visitors)}
        for r in by_hour.itertuples()
    ]
 
    # Hora de pico (mais visitantes em média)
    peak_hour = int(by_hour.loc[by_hour["visitors"].idxmax(), "hour_of_day"])
    quiet_hour = int(by_hour.loc[by_hour["visitors"].idxmin(), "hour_of_day"])
 
    # Dia mais e menos movimentado
    busiest_day   = by_day.loc[by_day["visitors"].idxmax()]
    quietest_day  = by_day.loc[by_day["visitors"].idxmin()]
 
    # Tempo médio de visita por pessoa (primeiro ao último evento)
    visit_duration = df.groupby("person_id")["entry_time"].agg(
        lambda x: (x.max() - x.min()).total_seconds()
    )
    avg_visit_min = safe_round(visit_duration.mean() / 60, 1)
 
    # Total de visitantes únicos na semana
    total_visitors = int(df["person_id"].nunique())
 
    return {
        "total_visitors_week": total_visitors,
        "visitors_by_day": visitors_by_day,
        "visitors_by_hour": visitors_by_hour,
        "peak_hour": peak_hour,
        "quiet_hour": quiet_hour,
        "busiest_day": {
            "date": str(busiest_day["visit_date"].date()),
            "day_name": busiest_day["day_name"],
            "visitors": int(busiest_day["visitors"]),
        },
        "quietest_day": {
            "date": str(quietest_day["visit_date"].date()),
            "day_name": quietest_day["day_name"],
            "visitors": int(quietest_day["visitors"]),
        },
        "avg_visit_duration_min": avg_visit_min,
    }
 
# 2. Métricas por zona
def compute_zone_metrics(df: pd.DataFrame) -> dict:
    # Tráfego por zona
    zone_traffic = (
        df.groupby("zone_id")["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "unique_visitors"})
    )
 
    # Dwell time médio (apenas linger — dwell_s > 0)
    linger_df = df[df["dwell_s"] > 0]
    dwell_by_zone = (
        linger_df.groupby("zone_id")["dwell_s"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "dwell_mean_s", "median": "dwell_median_s", "count": "linger_count"})
    )
 
    # Taxa de paragem: visitantes com linger / visitantes totais
    total_by_zone  = df.groupby("zone_id")["person_id"].nunique()
    linger_by_zone = linger_df.groupby("zone_id")["person_id"].nunique()
    stop_rate = (linger_by_zone / total_by_zone).fillna(0).reset_index()
    stop_rate.columns = ["zone_id", "stop_rate"]
 
    # Juntar tudo
    zone_df = (
        zone_traffic
        .merge(dwell_by_zone, on="zone_id", how="left")
        .merge(stop_rate,     on="zone_id", how="left")
    )
    zone_df["dwell_mean_s"]   = zone_df["dwell_mean_s"].apply(safe_round)
    zone_df["dwell_median_s"] = zone_df["dwell_median_s"].apply(safe_round)
    zone_df["stop_rate"]      = zone_df["stop_rate"].apply(lambda x: safe_round(x, 3))
    zone_df["linger_count"]   = zone_df["linger_count"].fillna(0).astype(int)
    zone_df["description"]    = zone_df["zone_id"].map(ZONE_DESCRIPTIONS)
 
    zones_list = zone_df.to_dict(orient="records")
 
    # Top-10 sequências de zonas mais frequentes
    sequences = []
    for pid, group in df.groupby("person_id"):
        zones = group.sort_values("entry_time")["zone_id"].tolist()
        for i in range(len(zones) - 1):
            sequences.append(f"{zones[i]} → {zones[i+1]}")
 
    seq_counts = Counter(sequences)
    top_sequences = [
        {"sequence": seq, "count": cnt}
        for seq, cnt in seq_counts.most_common(10)
    ]
 
    return {
        "by_zone": zones_list,
        "top_sequences": top_sequences,
    }
  
# 3. Funil de cliente
def compute_funnel(df: pd.DataFrame) -> dict:
    total = int(df["person_id"].nunique())
 
    # Visitantes que chegaram a cada tipo de zona
    reached_nav      = int(df[df["zone_id"].isin(NAV_ZONES)]["person_id"].nunique())
    reached_sections = int(df[df["zone_id"].isin(SECTION_ZONES)]["person_id"].nunique())
    reached_checkout = int(df[df["zone_id"].isin(CHECKOUT_ZONES)]["person_id"].nunique())
    reached_ck_exit  = int(df[df["zone_id"] == "Z_CK"]["person_id"].nunique())
 
    # Perfil de quem NÃO chegou à caixa
    no_checkout = df[~df["person_id"].isin(
        df[df["zone_id"].isin(CHECKOUT_ZONES)]["person_id"]
    )]
    no_checkout_profile = {}
    if len(no_checkout) > 0:
        nc_people = no_checkout.drop_duplicates("person_id")
        no_checkout_profile = {
            "gender_dist": nc_people["gender"].value_counts().to_dict(),
            "age_dist":    nc_people["age_range"].value_counts().to_dict(),
            "total":       int(nc_people["person_id"].nunique()),
        }
 
    def pct(n): return safe_round(n / total * 100, 1)
 
    return {
        "total_visitors": total,
        "reached_navigation": {"count": reached_nav,      "pct": pct(reached_nav)},
        "reached_sections":   {"count": reached_sections, "pct": pct(reached_sections)},
        "reached_checkout":   {"count": reached_checkout, "pct": pct(reached_checkout)},
        "completed_purchase":  {"count": reached_ck_exit,  "pct": pct(reached_ck_exit)},
        "no_checkout_profile": no_checkout_profile,
    }
 
# 4. Segmentação demográfica
def compute_demographics(df: pd.DataFrame) -> dict:
    people = df.drop_duplicates("person_id")[["person_id", "gender", "age_range"]]
 
    gender_dist  = people["gender"].value_counts().to_dict()
    age_dist     = people["age_range"].value_counts().to_dict()
 
    # Dwell time médio por segmento (género x zona)
    linger_df = df[df["dwell_s"] > 0]
    dwell_gender = (
        linger_df.groupby(["gender", "zone_id"])["dwell_s"]
        .mean()
        .reset_index()
        .rename(columns={"dwell_s": "avg_dwell_s"})
    )
    dwell_gender["avg_dwell_s"] = dwell_gender["avg_dwell_s"].apply(safe_round)
 
    # Distribuição de género por hora
    gender_by_hour = (
        df.drop_duplicates(["person_id", "hour_of_day"])
        .groupby(["hour_of_day", "gender"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "visitors"})
    )
 
    # Dwell time médio por faixa etária
    dwell_age = (
        linger_df.groupby("age_range")["dwell_s"]
        .mean()
        .apply(safe_round)
        .to_dict()
    )
 
    return {
        "gender_distribution":   gender_dist,
        "age_distribution":      age_dist,
        "dwell_by_gender_zone":  dwell_gender.to_dict(orient="records"),
        "gender_by_hour":        gender_by_hour.to_dict(orient="records"),
        "avg_dwell_by_age":      dwell_age,
    }
 
# 5. Deteção de anomalias (z-score, dia 7 vs dias 1-6)
def compute_anomalies(df: pd.DataFrame) -> dict:
    # Tráfego por zona × hora × dia
    traffic = (
        df.groupby(["day_num", "zone_id", "hour_of_day"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "visitors"})
    )
 
    baseline = traffic[traffic["day_num"] <= 6]
    day7     = traffic[traffic["day_num"] == 7]
 
    # Média e desvio padrão dos primeiros 6 dias por zona+hora
    stats = (
        baseline.groupby(["zone_id", "hour_of_day"])["visitors"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["std"] = stats["std"].fillna(0)
 
    # Comparar com dia 7
    merged = day7.merge(stats, on=["zone_id", "hour_of_day"], how="left")
    merged["std"]  = merged["std"].fillna(0)
    merged["mean"] = merged["mean"].fillna(0)
 
    # Z-score: quantos desvios padrão acima/abaixo da média
    merged["z_score"] = merged.apply(
        lambda r: (r["visitors"] - r["mean"]) / r["std"]
        if r["std"] > 0 else 0,
        axis=1
    )
 
    # Anomalias: |z| > 2
    anomalies = merged[merged["z_score"].abs() > 2].copy()
    anomalies = anomalies.sort_values("z_score", key=abs, ascending=False)
 
    anomalies_list = []
    for _, row in anomalies.iterrows():
        direction = "acima" if row["z_score"] > 0 else "abaixo"
        anomalies_list.append({
            "zone_id":      row["zone_id"],
            "zone_desc":    ZONE_DESCRIPTIONS.get(row["zone_id"], ""),
            "hour":         int(row["hour_of_day"]),
            "visitors_d7":  int(row["visitors"]),
            "baseline_mean": safe_round(row["mean"], 1),
            "baseline_std":  safe_round(row["std"], 1),
            "z_score":       safe_round(row["z_score"], 2),
            "direction":     direction,
            "description":   (
                f"Zona {row['zone_id']} às {int(row['hour_of_day'])}h teve "
                f"{int(row['visitors'])} visitantes no dia 7, vs média de "
                f"{safe_round(row['mean'], 1)} nos dias 1-6 "
                f"(z={safe_round(row['z_score'], 2)}, {direction} da média)"
            ),
        })
 
    return {
        "total_anomalies": len(anomalies_list),
        "anomalies": anomalies_list[:20],  # top 20 mais significativas
    }
 
# Main
def main():
    parser = argparse.ArgumentParser(description="Analytics — cálculo de métricas")
    parser.add_argument("--input",  required=True, help="Caminho para journeys.csv")
    parser.add_argument("--output", required=True, help="Caminho para metrics.json")
    args = parser.parse_args()
 
    print(f"[1/6] A carregar {args.input}")
    df = load_journeys(args.input)
    print(f"      {len(df):,} linhas, {df['person_id'].nunique():,} visitantes únicos.")
 
    print("[2/6] A calcular métricas de tráfego")
    traffic = compute_traffic(df)
 
    print("[3/6] A calcular métricas por zona")
    zones = compute_zone_metrics(df)
 
    print("[4/6] A calcular funil de cliente")
    funnel = compute_funnel(df)
 
    print("[5/6] A calcular segmentação demográfica")
    demographics = compute_demographics(df)
 
    print("[6/6] A detetar anomalias")
    anomalies = compute_anomalies(df)
 
    metrics = {
        "meta": {
            "generated_from": args.input,
            "week_start": str(df["visit_date"].min().date()),
            "week_end":   str(df["visit_date"].max().date()),
            "total_events": int(len(df)),
        },
        "traffic":      traffic,
        "zones":        zones,
        "funnel":       funnel,
        "demographics": demographics,
        "anomalies":    anomalies,
    }
 
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
 
    print(f"\nResumo")
    print(f"  Visitantes totais:        {traffic['total_visitors_week']:,}")
    print(f"  Dia mais movimentado:     {traffic['busiest_day']['day_name']} ({traffic['busiest_day']['visitors']} visitantes)")
    print(f"  Hora de pico:             {traffic['peak_hour']}h")
    print(f"  Taxa de conversão caixa:  {funnel['reached_checkout']['pct']}%")
    print(f"  Anomalias detetadas:      {anomalies['total_anomalies']}")
    print(f"\n  metrics.json escrito em {args.output}")
 
 
if __name__ == "__main__":
    main()