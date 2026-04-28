import pandas as pd

df = pd.read_csv('output/journeys.csv', parse_dates=['entry_time', 'exit_time'])

entry_zones        = {'Z_E1', 'Z_E2'}
exit_zones_strict  = {'Z_E1', 'Z_E2', 'Z_CK'}
exit_zones_relaxed = {'Z_E1', 'Z_E2', 'Z_CK', 'Z_C1', 'Z_C2', 'Z_C3'}

first_zones = df.sort_values('entry_time').groupby('person_id')['zone_id'].first()
last_zones  = df.sort_values('entry_time').groupby('person_id')['zone_id'].last()

total    = len(first_zones)
starts   = first_zones.isin(entry_zones).sum()
strict   = (first_zones.isin(entry_zones) & last_zones.isin(exit_zones_strict)).sum()
relaxed  = (first_zones.isin(entry_zones) & last_zones.isin(exit_zones_relaxed)).sum()

# consistência: pessoa em duas zonas ao mesmo tempo
violations = 0
for pid, group in df.groupby('person_id'):
    group = group.dropna(subset=['exit_time']).sort_values('entry_time')
    for i in range(len(group) - 1):
        if group.iloc[i+1]['entry_time'] < group.iloc[i]['exit_time']:
            violations += 1

zones_per_person = df.groupby('person_id')['zone_id'].count()

print(f"total trajectórias      : {total}")
print(f"começam em Z_E          : {starts} ({100*starts/total:.1f}%)")
print(f"completude estrita      : {strict} ({100*strict/total:.1f}%)")
print(f"completude relaxada     : {relaxed} ({100*relaxed/total:.1f}%)")
print(f"violações de sobreposição: {violations}")
print(f"zonas por pessoa — média: {zones_per_person.mean():.1f} | mediana: {zones_per_person.median():.0f} | max: {zones_per_person.max()}")