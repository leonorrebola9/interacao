import pandas as pd
import json
import argparse
from datetime import datetime, timedelta

class Stitcher:
    def __init__(self, zones_path):
        # Carrega o mapa de zonas para usar como restrição espacial
        with open(zones_path, 'r', encoding='utf-8') as f:
            self.zone_map = json.load(f)['zones']
        self.active_trajectories = {}  # person_id -> list of events
        self.person_count = 0
        self.max_gap = 300  # 5 minutos de gap máximo [cite: 71]

    def get_walk_time(self, from_zone, to_zone):
        """Retorna o tempo mínimo de caminhada entre zonas do zones.json."""
        if from_zone == to_zone:
            return 0
        return self.zone_map.get(from_zone, {}).get('walk_seconds', {}).get(to_zone, 15)

    def calculate_affinity(self, journey, event):
        """
        Calcula se um evento pode pertencer a uma trajetória.
        Retorna um score (0 a 1) ou -1 se for impossível.
        """
        last_event = journey[-1]
        time_diff = (event['timestamp'] - last_event['timestamp']).total_seconds()
        
        # 1. Consistência Temporal: não pode estar em dois sítios ao mesmo tempo
        if time_diff < 0:
            return -1
        
        # 2. Plausibilidade Física: tempo de caminhada do zones.json
        min_walk = self.get_walk_time(last_event['zone_id'], event['zone_id'])
        if time_diff < min_walk:
            return -1 # Teletransporte detetado!

        # 3. Gap Máximo: se passou muito tempo, a pessoa provavelmente saiu [cite: 71]
        if time_diff > self.max_gap:
            return -1

        # 4. Consistência de Atributos (com tolerância a erro de 8-12%) [cite: 39, 40, 72]
        score = 1.0
        if event['gender'] != last_event['gender']:
            score *= 0.5
        if event['age_range'] != last_event['age_range']:
            score *= 0.7
            
        return score

    def run(self, input_csv, output_csv):
        df = pd.read_csv(input_csv).sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        results = []
        
        for _, row in df.iterrows():
            event = row.to_dict()
            best_p_id = None
            best_score = -0.1
            
            # Tenta associar a alguém que já está na loja
            for p_id, journey in self.active_trajectories.items():
                score = self.calculate_affinity(journey, event)
                if score > best_score:
                    best_score = score
                    best_p_id = p_id
            
            # Se o score for baixo ou for uma nova entrada em Z_E 
            if best_score < 0.4:
                self.person_count += 1
                best_p_id = f"P_{self.person_count:04d}"
                self.active_trajectories[best_p_id] = [event]
            else:
                self.active_trajectories[best_p_id].append(event)
            
            # Se o evento for um 'exit' numa zona de saída, podemos "fechar" a trajetória 
            if event['event_type'] == 'exit' and (event['zone_id'].startswith('Z_E') or event['zone_id'] == 'Z_CK'):
                # Idealmente, moveríamos para completed_trajectories aqui para otimizar
                pass

        # Pós-processamento para o formato journeys.csv [cite: 78, 83]
        final_data = []
        for p_id, events in self.active_trajectories.items():
            # Agrupar eventos por zona para calcular dwell_s e tempos de entrada/saída
            zones_visited = {} # zone -> {entry, exit, linger_duration}
            
            for e in events:
                z = e['zone_id']
                if z not in zones_visited:
                    zones_visited[z] = {'entry': None, 'exit': None, 'dwell': 0, 'gender': e['gender'], 'age': e['age_range']}
                
                if e['event_type'] == 'entry':
                    zones_visited[z]['entry'] = e['timestamp']
                elif e['event_type'] == 'exit':
                    zones_visited[z]['exit'] = e['timestamp']
                elif e['event_type'] == 'linger':
                    zones_visited[z]['dwell'] = e['duration_s'] [cite: 38]

            for z_id, info in zones_visited.items():
                if info['entry'] and info['exit']:
                    final_data.append({
                        'person_id': p_id,
                        'zone_id': z_id,
                        'entry_time': info['entry'],
                        'exit_time': info['exit'],
                        'dwell_s': info['dwell'],
                        'gender': info['gender'],
                        'age_range': info['age'],
                        'visit_date': info['entry'].date(),
                        'hour_of_day': info['entry'].hour
                    })

        pd.DataFrame(final_data).to_csv(output_csv, index=False)
        print(f"Sucesso: {output_csv} gerado com {self.person_count} trajetórias.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # O zones.json deve estar na raiz ou pasta data
    stitcher = Stitcher("zones.json")
    stitcher.run(args.input, args.output)