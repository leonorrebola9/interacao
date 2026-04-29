import argparse
import json
from collections import Counter
from datetime import datetime
import pandas as pd

days_pt = {
    0: "Segunda",
    1: "Terça",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sábado",
    6: "Domingo",
}

entry_zones   = {"Z_E1", "Z_E2"}
checkout      = {"Z_C1", "Z_C2", "Z_C3"}
checkout_exit = {"Z_CK"}
exit_zones    = {"Z_E1", "Z_E2", "Z_CK"}

# ─── Carregamento ─────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    df["visit_date"]  = pd.to_datetime(df["visit_date"]).dt.date
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    return df

# ─── 1. Métricas de tráfego ───────────────────────────────────────────────────

def traffic_metrics(df: pd.DataFrame) -> dict:
    # Visitantes únicos por dia
    by_day = (
        df.groupby("visit_date")["person_id"]
        .nunique()
        .reset_index()
    )
    by_day["day_name"] = pd.to_datetime(by_day["visit_date"]).dt.dayofweek.map(days_pt)

    # Média de visitantes por hora (média entre os 7 dias)
    by_hour = (
        df[df["zone_id"].isin(entry_zones)]
        .groupby(["visit_date", "hour_of_day"])["person_id"]
        .nunique()
        .groupby("hour_of_day")
        .mean()
        .round(1)
    )

    # Duração total de visita por pessoa por dia
    durations = []
    for (pid, date), grp in df.groupby(["person_id", "visit_date"]):
        start = grp["entry_time"].min()
        end   = grp["exit_time"].max()
        if pd.notna(start) and pd.notna(end):
            dur = (end - start).total_seconds()
            if 30 < dur < 7200:
                durations.append(dur)

    avg_s = round(sum(durations) / max(len(durations), 1), 1)

    busiest = by_day.loc[by_day["person_id"].idxmax()]
    quietest = by_day.loc[by_day["person_id"].idxmin()]

    return {
        "total_unique_visitors": int(df["person_id"].nunique()),
        "visitors_by_day": {
            str(r["visit_date"]): {
                "count":    int(r["person_id"]),
                "day_name": r["day_name"],
            }
            for _, r in by_day.iterrows()
        },
        "avg_visitors_by_hour": {
            str(h): v for h, v in by_hour.items()
        },
        "avg_visit_duration_s":   avg_s,
        "avg_visit_duration_min": round(avg_s / 60, 1),
        "busiest_day": {
            "date":     str(busiest["visit_date"]),
            "day_name": busiest["day_name"],
            "count":    int(busiest["person_id"]),
        },
        "quietest_day": {
            "date":     str(quietest["visit_date"]),
            "day_name": quietest["day_name"],
            "count":    int(quietest["person_id"]),
        },
    }

# ─── 2. Métricas por zona ─────────────────────────────────────────────────────

def zone_metrics(df: pd.DataFrame) -> dict:
    by_zone = {}
    for zone_id, grp in df.groupby("zone_id"):
        total_visitors  = int(grp["person_id"].nunique())
        linger_rows     = grp[grp["dwell_s"] > 0]
        avg_dwell       = round(float(linger_rows["dwell_s"].mean()), 1) if len(linger_rows) else 0.0
        visitors_linger = int(linger_rows["person_id"].nunique())
        stop_rate       = round(visitors_linger / max(total_visitors, 1), 3)

        by_zone[zone_id] = {
            "total_visitors": total_visitors,
            "avg_dwell_s":    avg_dwell,
            "stop_rate":      stop_rate,
        }

    # Top 10 sequências de zonas consecutivas
    seq_counter: Counter = Counter()
    for pid, grp in df.groupby("person_id"):
        zones = grp.sort_values("entry_time")["zone_id"].tolist()
        for i in range(len(zones) - 1):
            seq_counter[(zones[i], zones[i + 1])] += 1

    top_sequences = [
        {"from": a, "to": b, "count": c}
        for (a, b), c in seq_counter.most_common(10)
    ]

    # Top 5 zonas com mais tráfego e menos tráfego
    sorted_zones = sorted(by_zone.items(), key=lambda x: x[1]["total_visitors"], reverse=True)
    top5    = [{"zone_id": z, **v} for z, v in sorted_zones[:5]]
    bottom5 = [{"zone_id": z, **v} for z, v in sorted_zones[-5:]]

    return {
        "by_zone":        by_zone,
        "top_5_traffic":  top5,
        "bottom_5_traffic": bottom5,
        "top_10_sequences": top_sequences,
    }

# ─── 3. Funil de cliente ──────────────────────────────────────────────────────

def funnel_metrics(df: pd.DataFrame) -> dict:
    all_visitors     = set(df["person_id"].unique())
    entered          = set(df[df["zone_id"].isin(entry_zones)]["person_id"])
    reached_nav      = set(df[df["zone_id"].str.startswith("Z_N")]["person_id"])
    reached_sections = set(df[df["zone_id"].str.startswith("Z_S")]["person_id"])
    reached_checkout = set(df[df["zone_id"].isin(checkout)]["person_id"])
    exited           = set(df[df["zone_id"].isin(exit_zones)]["person_id"])

    no_checkout = all_visitors - reached_checkout
    nc_df = df[df["person_id"].isin(no_checkout)].drop_duplicates("person_id")

    def pct(a, b):
        return round(100 * len(a) / max(len(b), 1), 1)

    return {
        "total_visitors":        len(all_visitors),
        "entered_via_door":      len(entered),
        "reached_navigation":    len(reached_nav),
        "reached_sections":      len(reached_sections),
        "reached_checkout":      len(reached_checkout),
        "exited_correctly":      len(exited),
        "conversion_rate_pct":   pct(reached_checkout, all_visitors),
        "no_checkout_count":     len(no_checkout),
        "no_checkout_pct":       pct(no_checkout, all_visitors),
        "no_checkout_profile": {
            "gender": nc_df["gender"].value_counts(normalize=True).round(3).to_dict(),
            "age":    nc_df["age_range"].value_counts(normalize=True).round(3).to_dict(),
        },
    }

# ─── 4. Segmentação demográfica ───────────────────────────────────────────────

def demographic_metrics(df: pd.DataFrame) -> dict:
    # Dwell time médio por género e zona (top 5 zonas)
    top_zones = (
        df[df["dwell_s"] > 0]
        .groupby("zone_id")["dwell_s"]
        .mean()
        .nlargest(5)
        .index.tolist()
    )
    dwell_by_gender = {}
    for gender, grp in df[df["zone_id"].isin(top_zones) & (df["dwell_s"] > 0)].groupby("gender"):
        dwell_by_gender[gender] = (
            grp.groupby("zone_id")["dwell_s"]
            .mean()
            .round(1)
            .to_dict()
        )

    # Distribuição de faixa etária por hora (nas entradas)
    entry_df = df[df["zone_id"].isin(entry_zones)]
    age_by_hour = {}
    for hour, grp in entry_df.groupby("hour_of_day"):
        age_by_hour[str(hour)] = (
            grp.drop_duplicates("person_id")["age_range"]
            .value_counts(normalize=True)
            .round(3)
            .to_dict()
        )

    # Visitantes por segmento (género × faixa etária)
    segment = (
        df.drop_duplicates("person_id")
        .groupby(["gender", "age_range"])
        .size()
        .reset_index(name="count")
    )
    segments = [
        {"gender": r["gender"], "age_range": r["age_range"], "count": int(r["count"])}
        for _, r in segment.iterrows()
    ]

    return {
        "avg_dwell_by_gender_top_zones": dwell_by_gender,
        "age_distribution_by_hour":      age_by_hour,
        "visitor_segments":              segments,
    }

# ─── 5. Detecção de anomalias ─────────────────────────────────────────────────

def anomaly_detection(df: pd.DataFrame) -> dict:
    """
    Para cada zona + hora, calcula média e desvio padrão nos primeiros 6 dias.
    Identifica onde o dia 7 se desvia mais de 2σ.
    """
    dates = sorted(df["visit_date"].unique())
    if len(dates) < 2:
        return {"anomalies": [], "note": "dados insuficientes"}

    baseline_dates = dates[:6]
    test_date      = dates[-1]

    counts = (
        df.groupby(["visit_date", "zone_id", "hour_of_day"])["person_id"]
        .nunique()
        .reset_index()
        .rename(columns={"person_id": "visitors"})
    )

    baseline = counts[counts["visit_date"].isin(baseline_dates)]
    test     = counts[counts["visit_date"] == test_date]

    stats = (
        baseline.groupby(["zone_id", "hour_of_day"])["visitors"]
        .agg(["mean", "std"])
        .fillna(0)
        .reset_index()
    )
    stats.columns = ["zone_id", "hour_of_day", "mean", "std"]

    merged = test.merge(stats, on=["zone_id", "hour_of_day"], how="left")
    merged["std"]     = merged["std"].fillna(1).clip(lower=1)
    merged["mean"]    = merged["mean"].fillna(0)
    merged["z_score"] = (merged["visitors"] - merged["mean"]) / merged["std"]

    anomalies_df = (
        merged[merged["z_score"].abs() > 2]
        .sort_values("z_score", key=abs, ascending=False)
    )

    return {
        "test_date":      str(test_date),
        "baseline_dates": [str(d) for d in baseline_dates],
        "total_anomalies": len(anomalies_df),
        "anomalies": [
            {
                "zone_id":   row["zone_id"],
                "hour":      int(row["hour_of_day"]),
                "observed":  int(row["visitors"]),
                "expected":  round(float(row["mean"]), 1),
                "z_score":   round(float(row["z_score"]), 2),
                "direction": "acima" if row["z_score"] > 0 else "abaixo",
            }
            for _, row in anomalies_df.iterrows()
        ],
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="output/journeys.csv", help="CSV de trajectórias")
    parser.add_argument("--output", default="output/metrics.json", help="JSON de métricas")
    args = parser.parse_args()

    print(f"[1/5] A carregar '{args.input}'")
    df = load(args.input)
    print(f"      {len(df)} linhas, {df['person_id'].nunique()} pessoas, {df['visit_date'].nunique()} dias.")

    print("[2/5] Métricas de tráfego")
    traffic = traffic_metrics(df)

    print("[3/5] Métricas por zona")
    zones = zone_metrics(df)

    print("[4/5] Funil e segmentação demográfica")
    funnel = funnel_metrics(df)
    demo   = demographic_metrics(df)

    print("[5/5] Detecção de anomalias")
    anomalies = anomaly_detection(df)

    metrics = {
        "generated_at": datetime.now().isoformat(),
        "traffic":      traffic,
        "zones":        zones,
        "funnel":       funnel,
        "demographics": demo,
        "anomalies":    anomalies,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nmetrics.json escrito em '{args.output}'")
    print(f"  Visitantes únicos  : {traffic['total_unique_visitors']}")
    print(f"  Dia mais movimentado: {traffic['busiest_day']['day_name']} ({traffic['busiest_day']['count']} visitantes)")
    print(f"  Conversão p/ caixa : {funnel['conversion_rate_pct']}%")
    print(f"  Anomalias (dia 7)  : {anomalies['total_anomalies']}")

if __name__ == "__main__":
    main()