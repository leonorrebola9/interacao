import pandas as pd
import json
import argparse
import os
from datetime import datetime

class Stitcher:
    def __init__(self, zones_path):
        with open(zones_path, 'r', encoding='utf-8') as f:
            self.zone_map = json.load(f)['zones']
        self.active_trajectories = {}  # person_id -> list of events
        self.completed_trajectories = [] # Guardar aqui para libertar a busca
        self.person_count = 0
        self.max_gap = 300  # 5 minutos 

    def get_walk_time(self, from_zone, to_zone):
        if from_zone == to_zone: return 0
        return self.zone_map.get(from_zone, {}).get('walk_seconds', {}).get(to_zone, 15)

    def calculate_affinity(self, journey, event):
        last_event = journey[-1]
        # O dataset está ordenado, mas garantimos a consistência temporal [cite: 68]
        time_diff = (event['timestamp'] - last_event['timestamp']).total_seconds()
        
        if time_diff < 0: return -1
        
        # Validação espacial usando o zones.json
        min_walk = self.get_walk_time(last_event['zone_id'], event['zone_id'])
        if time_diff < min_walk: return -1

        if time_diff > self.max_gap: return -1

        # Score baseado em atributos [cite: 72]
        score = 1.0
        if event['gender'] != last_event['gender']: score *= 0.5
        if event['age_range'] != last_event['age_range']: score *= 0.7
        return score

    def run(self, input_csv, output_csv):
        df = pd.read_csv(input_csv).sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Otimização: Converter para lista de dicts (muito mais rápido que iterrows)
        events_list = df.to_dict('records')
        total = len(events_list)

        print(f"A processar {total} eventos")
        
        for i, event in enumerate(events_list):
            if i % 25000 == 0:
                print(f"Progresso: {i/total*100:.1f}%...")

            current_time = event['timestamp']
            best_p_id = None
            best_score = -0.1
            
            # OTIMIZAÇÃO: Limpar trajetórias inativas para manter a busca rápida
            inactive_ids = [
                p_id for p_id, j in self.active_trajectories.items() 
                if (current_time - j[-1]['timestamp']).total_seconds() > self.max_gap
            ]
            for p_id in inactive_ids:
                self.completed_trajectories.append((p_id, self.active_trajectories.pop(p_id)))

            # Matchmaking apenas com quem ainda está na loja
            for p_id, journey in self.active_trajectories.items():
                score = self.calculate_affinity(journey, event)
                if score > best_score:
                    best_score = score
                    best_p_id = p_id
            
            if best_score < 0.4:
                self.person_count += 1
                new_id = f"P_{self.person_count:04d}"
                self.active_trajectories[new_id] = [event]
            else:
                self.active_trajectories[best_p_id].append(event)

        # Mover o que restou nas ativas para as completas
        for p_id, journey in self.active_trajectories.items():
            self.completed_trajectories.append((p_id, journey))

        self.save_journeys(output_csv)

    def save_journeys(self, output_csv):
        final_data = []
        for p_id, events in self.completed_trajectories:
            # Agrupar por zona para cumprir o schema do journeys.csv [cite: 78, 83]
            zones_visited = {}
            for e in events:
                z = e['zone_id']
                if z not in zones_visited:
                    zones_visited[z] = {'entry': None, 'exit': None, 'dwell': 0, 'g': e['gender'], 'a': e['age_range']}
                
                if e['event_type'] == 'entry': zones_visited[z]['entry'] = e['timestamp']
                elif e['event_type'] == 'exit': zones_visited[z]['exit'] = e['timestamp']
                elif e['event_type'] == 'linger': zones_visited[z]['dwell'] = e['duration_s']

            for z_id, info in zones_visited.items():
                if info['entry']: # Só regista se tiver pelo menos o entry
                    final_data.append({
                        'person_id': p_id,
                        'zone_id': z_id,
                        'entry_time': info['entry'],
                        'exit_time': info['exit'] if info['exit'] else info['entry'],
                        'dwell_s': info['dwell'],
                        'gender': info['g'],
                        'age_range': info['a'],
                        'visit_date': info['entry'].date(),
                        'hour_of_day': info['entry'].hour # Inteiro 0-23 
                    })

        pd.DataFrame(final_data).to_csv(output_csv, index=False)
        print(f"Concluído! {self.person_count} trajetórias reconstruídas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    z_path = "zones.json"
    if not os.path.exists(z_path):
        z_path = os.path.join("..", "zones.json")

    stitcher = Stitcher(z_path)
    stitcher.run(args.input, args.output)