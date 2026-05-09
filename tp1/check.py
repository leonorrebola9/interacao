import pandas as pd

df = pd.read_csv("output/journeys.csv", parse_dates=["entry_time", "exit_time"])

# Para cada trajetória, ver a sequência de zonas
sequences = df.sort_values(["person_id", "entry_time"]).groupby("person_id")["zone_id"].apply(list)

# Contar trajetórias que vão de Z_E diretamente para Z_C
direct_to_checkout = 0
total = 0
for pid, zones in sequences.items():
    total += 1
    if len(zones) >= 2:
        if zones[0] in {"Z_E1", "Z_E2"} and zones[1] in {"Z_C1", "Z_C2", "Z_C3", "Z_CK"}:
            direct_to_checkout += 1

print(f"Total trajetórias: {total}")
print(f"Z_E → Z_C direto: {direct_to_checkout} ({direct_to_checkout/total*100:.1f}%)")
print(f"\nZona mais frequente como 2ª zona:")
second_zones = sequences[sequences.apply(len) >= 2].apply(lambda x: x[1])
print(second_zones.value_counts().head(10))

print(f"\nDistribuição por número de zonas visitadas:")
zone_counts = sequences.apply(len).value_counts().sort_index()
for n, count in zone_counts.items():
    print(f"  {n} zona(s): {count} trajetórias ({count/total*100:.1f}%)")