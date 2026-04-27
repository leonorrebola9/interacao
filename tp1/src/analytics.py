import pandas as pd
import json
import argparse
import os

def calculate_analytics(input_csv, output_json):
    # Carregar os dados gerados pelo stitcher
    df = pd.read_csv(input_csv)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # 1. Métricas de Tráfego Geral
    total_visitors = int(df['person_id'].nunique())
    avg_visit_duration = float((df['exit_time'] - df['entry_time']).dt.total_seconds().mean())
    
    # 2. Métricas por Zona 
    zone_metrics = {}
    for zone in df['zone_id'].unique():
        zone_df = df[df['zone_id'] == zone]
        visitors_in_zone = int(zone_df['person_id'].nunique())
        # Taxa de paragem: visitantes com linger > 0 / visitantes totais na zona
        stoppers = int(zone_df[zone_df['dwell_s'] > 0]['person_id'].nunique())
        
        zone_metrics[zone] = {
            "total_visitors": visitors_in_zone,
            "avg_dwell_time": float(zone_df[zone_df['dwell_s'] > 0]['dwell_s'].mean() if stoppers > 0 else 0),
            "stop_rate": float(stoppers / visitors_in_zone if visitors_in_zone > 0 else 0)
        }

    # 3. Deteção de Anomalias (Dia 7 vs Média Dias 1-6) 
    # O dataset cobre 7 dias consecutivos
    df['day_index'] = df['entry_time'].dt.dayofyear
    days = sorted(df['day_index'].unique())
    
    anomalies = []
    if len(days) >= 7:
        day_7 = days[-1]
        historical_df = df[df['day_index'] < day_7]
        
        # Média de tráfego por zona/hora nos primeiros 6 dias
        history_stats = historical_df.groupby(['zone_id', 'hour_of_day'])['person_id'].nunique().groupby(['zone_id', 'hour_of_day']).mean().reset_index()
        
        # Tráfego no dia 7
        day_7_traffic = df[df['day_index'] == day_7].groupby(['zone_id', 'hour_of_day'])['person_id'].nunique().reset_index()
        
        for _, row in day_7_traffic.iterrows():
            hist_match = history_stats[(history_stats['zone_id'] == row['zone_id']) & (history_stats['hour_of_day'] == row['hour_of_day'])]
            if not hist_match.empty:
                avg = hist_match.iloc[0]['person_id']
                # Desvio > 20% da média
                diff = abs(row['person_id'] - avg)
                if diff > (avg * 0.20):
                    anomalies.append({
                        "zone_id": row['zone_id'],
                        "hour": int(row['hour_of_day']),
                        "observed": int(row['person_id']),
                        "expected": round(float(avg), 2),
                        "deviation_pct": round(float((row['person_id'] - avg) / avg * 100), 2)
                    })

    # 4. Funil de Cliente e Conversão
    # Zonas de checkout: Z_C1, Z_C2, Z_C3
    checkout_zones = ['Z_C1', 'Z_C2', 'Z_C3']
    visitors_at_checkout = df[df['zone_id'].isin(checkout_zones)]['person_id'].nunique()
    conversion_rate = float(visitors_at_checkout / total_visitors if total_visitors > 0 else 0)

    # 5. Segmentação Demográfica
    demo_dist = df.groupby(['gender', 'age_range'])['person_id'].nunique().to_dict()

    # Estrutura final do metrics.json
    metrics = {
        "store_summary": {
            "total_visitors": total_visitors,
            "avg_visit_duration_seconds": round(avg_visit_duration, 2),
            "conversion_rate": round(conversion_rate, 4)
        },
        "zone_analysis": zone_metrics,
        "anomalies": anomalies,
        "demographics": {f"{k[0]}_{k[1]}": int(v) for k, v in demo_dist.items()},
        "top_sequences": [list(x) for x in df.groupby('person_id')['zone_id'].apply(list).value_counts().head(10).index]
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    print(f"Analytics concluído. Métricas guardadas em: {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    calculate_analytics(args.input, args.output)