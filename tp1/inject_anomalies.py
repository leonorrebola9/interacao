import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

# Seed para reprodutibilidade
np.random.seed(42)

def inject_anomalies(input_path: str, output_path: str):
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    original_len = len(df)

    print(f"Dataset original: {original_len:,} eventos")
    print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    anomalies_injected = []

    # ─────────────────────────────────────────────
    # Anomalia 1: Queda para zero em Z_S3 no dia 7 entre 14h-15h
    # Simula obstrução física ou encerramento temporário
    # ─────────────────────────────────────────────
    day7 = df["timestamp"].dt.date == df["timestamp"].dt.date.unique()[-1]
    zone_s3 = df["zone_id"] == "Z_S3"
    hour_14 = df["timestamp"].dt.hour == 14

    mask_a1 = day7 & zone_s3 & hour_14
    removed_a1 = mask_a1.sum()
    df = df[~mask_a1].copy()

    anomalies_injected.append({
        "id": "ANO_001",
        "tipo": "queda_zero",
        "zona": "Z_S3",
        "hora": 14,
        "dia": "dia 7",
        "descricao": f"Removidos {removed_a1} eventos — simula encerramento temporário",
    })
    print(f"\n[ANO_001] Removidos {removed_a1} eventos de Z_S3 às 14h do dia 7")

    # ─────────────────────────────────────────────
    # Anomalia 2: Pico de tráfego em Z_N5 no dia 7 às 11h
    # Simula evento promocional não registado
    # ─────────────────────────────────────────────
    zone_n5 = df["zone_id"] == "Z_N5"
    hour_11 = df["timestamp"].dt.hour == 11
    day6 = df["timestamp"].dt.date == df["timestamp"].dt.date.unique()[-2]

    # Pegar eventos de Z_N5 às 11h do dia 6 e duplicar no dia 7
    source = df[day6 & zone_n5 & hour_11].copy()
    if len(source) > 0:
        duplicated = source.copy()
        # Mover para o dia 7
        duplicated["timestamp"] = duplicated["timestamp"] + pd.Timedelta(days=1)
        # Novos event_ids
        duplicated["event_id"] = [f"INJ_{i:04d}" for i in range(len(duplicated))]
        df = pd.concat([df, duplicated], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        anomalies_injected.append({
            "id": "ANO_002",
            "tipo": "pico_trafego",
            "zona": "Z_N5",
            "hora": 11,
            "dia": "dia 7",
            "descricao": f"Duplicados {len(duplicated)} eventos — simula pico promocional",
        })
        print(f"[ANO_002] Duplicados {len(duplicated)} eventos em Z_N5 às 11h do dia 7")

    # ─────────────────────────────────────────────
    # Anomalia 3: Queda abrupta em Z_C2 no dia 7 às 17h (hora de pico)
    # Simula avaria numa caixa registadora
    # ─────────────────────────────────────────────
    zone_c2 = df["zone_id"] == "Z_C2"
    hour_17 = df["timestamp"].dt.hour == 17
    day7_new = df["timestamp"].dt.date == df["timestamp"].dt.date.unique()[-1]

    mask_a3 = day7_new & zone_c2 & hour_17
    # Remover 80% dos eventos
    idx_to_remove = df[mask_a3].sample(frac=0.8, random_state=42).index
    removed_a3 = len(idx_to_remove)
    df = df.drop(idx_to_remove).reset_index(drop=True)

    anomalies_injected.append({
        "id": "ANO_003",
        "tipo": "queda_parcial",
        "zona": "Z_C2",
        "hora": 17,
        "dia": "dia 7",
        "descricao": f"Removidos {removed_a3} eventos (80%) — simula avaria de caixa",
    })
    print(f"[ANO_003] Removidos {removed_a3} eventos de Z_C2 às 17h do dia 7 (80%)")

    # ─────────────────────────────────────────────
    # Guardar resultado
    # ─────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nDataset com anomalias: {len(df):,} eventos")
    print(f"Diferença: {len(df) - original_len:+,} eventos")
    print(f"\nOutput: {output_path}")

    # Guardar registo das anomalias injetadas
    import json
    anomalies_log = Path(output_path).parent / "injected_anomalies.json"
    with open(anomalies_log, "w", encoding="utf-8") as f:
        json.dump(anomalies_injected, f, ensure_ascii=False, indent=2)
    print(f"Registo: {anomalies_log}")

    return anomalies_injected


if __name__ == "__main__":
    inject_anomalies(
        input_path="data/events.csv",
        output_path="data/events_with_anomalies.csv",
    )