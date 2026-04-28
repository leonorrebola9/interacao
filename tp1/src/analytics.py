import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone


def load(path):
    df = pd.read_csv(path, parse_dates=['entry_time', 'exit_time', 'visit_date'])
    df['hour_of_day'] = df['hour_of_day'].astype(int)
    df['dwell_s']     = df['dwell_s'].fillna(0).astype(float)
    return df


# ── 1. tráfego ────────────────────────────────────────────────────────────────

def traffic(df):
    visitors_per_day = (
        df.groupby(df['visit_date'].dt.date)['person_id']
        .nunique().sort_index()
    )
    visitors_per_hour = (
        df.groupby('hour_of_day')['person_id']
        .nunique().sort_index()
    )
    df['weekday'] = df['visit_date'].dt.day_name()
    visitors_per_weekday = (
        df.groupby('weekday')['person_id'].nunique().to_dict()
    )

    duration = (
        df.groupby('person_id')
        .agg(first=('entry_time', 'min'), last=('exit_time', 'max'))
        .assign(dur=lambda x: (x['last'] - x['first']).dt.total_seconds())
    )

    return {
        'total_unique_visitors'        : int(df['person_id'].nunique()),
        'visitors_per_day'             : {str(k): int(v) for k, v in visitors_per_day.items()},
        'visitors_per_hour'            : {str(k): int(v) for k, v in visitors_per_hour.items()},
        'visitors_per_weekday'         : visitors_per_weekday,
        'peak_hour'                    : int(visitors_per_hour.idxmax()),
        'busiest_day'                  : str(visitors_per_day.idxmax()),
        'quietest_day'                 : str(visitors_per_day.idxmin()),
        'avg_visit_duration_minutes'   : round(duration['dur'].mean() / 60, 2),
        'median_visit_duration_minutes': round(duration['dur'].median() / 60, 2),
    }


# ── 2. zonas ─────────────────────────────────────────────────────────────────

def zones(df):
    zone_traffic = df.groupby('zone_id')['person_id'].count().sort_values(ascending=False)
    zone_unique  = df.groupby('zone_id')['person_id'].nunique()

    dwell_df     = df[df['dwell_s'] > 0]
    zone_dwell   = dwell_df.groupby('zone_id')['dwell_s'].mean()

    linger_visitors = df[df['dwell_s'] > 0].groupby('zone_id')['person_id'].nunique()
    total_visitors  = df.groupby('zone_id')['person_id'].nunique()
    stop_rate       = (linger_visitors / total_visitors).fillna(0)

    # sequências de zonas consecutivas por pessoa (bigrams)
    seq_counts = {}
    for pid, group in df.groupby('person_id'):
        zones_visited = group.sort_values('entry_time')['zone_id'].tolist()
        for a, b in zip(zones_visited, zones_visited[1:]):
            if a == b:
                continue
            key = f"{a}→{b}"
            seq_counts[key] = seq_counts.get(key, 0) + 1
    top_sequences = sorted(seq_counts.items(), key=lambda x: -x[1])[:10]

    zone_stats = {}
    for zone in zone_traffic.index:
        zone_stats[zone] = {
            'total_visits'   : int(zone_traffic.get(zone, 0)),
            'unique_visitors': int(zone_unique.get(zone, 0)),
            'avg_dwell_s'    : round(float(zone_dwell.get(zone, 0)), 1),
            'stop_rate'      : round(float(stop_rate.get(zone, 0)), 3),
        }

    return {
        'zone_stats'       : zone_stats,
        'top_zones_traffic': [{'zone': z, 'visits': int(v)} for z, v in zone_traffic.head(10).items()],
        'top_zones_dwell'  : [{'zone': z, 'avg_dwell_s': round(float(zone_dwell.get(z, 0)), 1)}
                               for z in zone_dwell.sort_values(ascending=False).head(10).index],
        'top_sequences'    : [{'sequence': k, 'count': v} for k, v in top_sequences],
    }


# ── 3. funil ─────────────────────────────────────────────────────────────────

def funnel(df):
    entry_z    = {'Z_E1', 'Z_E2'}
    checkout_z = {'Z_C1', 'Z_C2', 'Z_C3'}
    ck_z       = {'Z_CK'}
    nav_z      = {z for z in df['zone_id'].unique() if z.startswith('Z_N')}
    section_z  = {z for z in df['zone_id'].unique() if z.startswith('Z_S')}

    all_pids      = set(df['person_id'].unique())
    entered       = set(df[df['zone_id'].isin(entry_z)]['person_id'])
    reached_nav   = set(df[df['zone_id'].isin(nav_z)]['person_id'])
    reached_sec   = set(df[df['zone_id'].isin(section_z)]['person_id'])
    reached_ck    = set(df[df['zone_id'].isin(checkout_z)]['person_id'])
    exited        = set(df[df['zone_id'].isin(ck_z)]['person_id'])

    def pct(a, b):
        return round(100 * len(a) / len(b), 1) if b else 0.0

    non_converters    = entered - reached_ck
    nc_df             = df[df['person_id'].isin(non_converters)].drop_duplicates('person_id')
    nc_gender         = nc_df['gender'].value_counts(normalize=True).round(3).to_dict()
    nc_age            = nc_df['age_range'].value_counts(normalize=True).round(3).to_dict()
    last_zone_nc      = (
        df[df['person_id'].isin(non_converters)]
        .sort_values('entry_time')
        .groupby('person_id')['zone_id'].last()
        .value_counts().head(5).to_dict()
    )

    return {
        'funnel': {
            'entered'           : len(entered),
            'reached_navigation': len(reached_nav & entered),
            'reached_sections'  : len(reached_sec & entered),
            'reached_checkout'  : len(reached_ck & entered),
            'exited_via_ck'     : len(exited & entered),
        },
        'conversion_rates_pct': {
            'entry_to_navigation': pct(reached_nav & entered, entered),
            'entry_to_sections'  : pct(reached_sec & entered, entered),
            'entry_to_checkout'  : pct(reached_ck & entered, entered),
        },
        'non_converters': {
            'count'                : len(non_converters),
            'pct_of_entered'       : pct(non_converters, entered),
            'gender_distribution'  : nc_gender,
            'age_distribution'     : nc_age,
            'last_zone_before_exit': {k: int(v) for k, v in last_zone_nc.items()},
        },
    }


# ── 4. demografia ─────────────────────────────────────────────────────────────

def demographics(df):
    gender_overall = (
        df.drop_duplicates('person_id')['gender']
        .value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    age_overall = (
        df.drop_duplicates('person_id')['age_range']
        .value_counts(normalize=True).mul(100).round(1).to_dict()
    )
    gender_by_hour = (
        df.groupby(['hour_of_day', 'gender'])['person_id'].nunique()
        .unstack(fill_value=0)
        .to_dict(orient='index')
    )
    gender_by_hour = {str(h): {g: int(v) for g, v in vals.items()}
                      for h, vals in gender_by_hour.items()}

    age_by_hour = (
        df.groupby(['hour_of_day', 'age_range'])['person_id'].nunique()
        .unstack(fill_value=0)
        .to_dict(orient='index')
    )
    age_by_hour = {str(h): {a: int(v) for a, v in vals.items()}
                   for h, vals in age_by_hour.items()}

    return {
        'gender_overall_pct': gender_overall,
        'age_overall_pct'   : age_overall,
        'gender_by_hour'    : gender_by_hour,
        'age_by_hour'       : age_by_hour,
    }


# ── 5. anomalias ──────────────────────────────────────────────────────────────

def anomalies(df):
    df = df.copy()
    df['date']    = df['visit_date'].dt.date
    min_date      = df['date'].min()
    df['day_num'] = df['date'].apply(lambda d: (d - min_date).days + 1)

    traffic = (
        df.groupby(['day_num', 'zone_id', 'hour_of_day'])['person_id']
        .nunique().reset_index(name='visitors')
    )

    baseline = traffic[traffic['day_num'] <= 6]
    day7     = traffic[traffic['day_num'] == 7]

    if day7.empty:
        return {'note': 'dia 7 não encontrado', 'anomalies': []}

    stats = (
        baseline.groupby(['zone_id', 'hour_of_day'])['visitors']
        .agg(mean='mean', std='std').fillna(0).reset_index()
    )

    found = []
    for _, row in day7.iterrows():
        s = stats[(stats['zone_id'] == row['zone_id']) &
                  (stats['hour_of_day'] == row['hour_of_day'])]
        if s.empty:
            continue
        mu, sig = s['mean'].values[0], s['std'].values[0]
        if sig < 1e-6:
            continue
        z = (row['visitors'] - mu) / sig
        if abs(z) >= 2.0:
            found.append({
                'zone'          : row['zone_id'],
                'hour'          : int(row['hour_of_day']),
                'day7_visitors' : int(row['visitors']),
                'baseline_mean' : round(float(mu), 1),
                'baseline_std'  : round(float(sig), 1),
                'z_score'       : round(float(z), 2),
                'direction'     : 'above' if z > 0 else 'below',
            })

    found.sort(key=lambda x: -abs(x['z_score']))
    return {
        'total'    : len(found),
        'threshold': 2.0,
        'anomalies': found,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    print(f"a carregar {args.input}")
    df = load(args.input)
    print(f"  {len(df):,} linhas | {df['person_id'].nunique():,} pessoas")

    print("a calcular métricas")
    metrics = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'data_period' : {
            'start': str(df['visit_date'].min().date()),
            'end'  : str(df['visit_date'].max().date()),
            'days' : int(df['visit_date'].dt.date.nunique()),
        },
        'traffic'     : traffic(df),
        'zones'        : zones(df),
        'funnel'       : funnel(df),
        'demographics' : demographics(df),
        'anomalies'    : anomalies(df),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(f"guardado: {args.output}")
    print(f"  anomalias detectadas: {metrics['anomalies']['total']}")


if __name__ == '__main__':
    main()