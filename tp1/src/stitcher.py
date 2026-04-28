import pandas as pd
import numpy as np
import json
import argparse
import os
from datetime import datetime, timedelta
from scipy.optimize import linear_sum_assignment
from heapq import heappush, heappop

# ── Constantes ─────────────────────────────────────────────────────────────
max_gap_s       = 300    # 5 min: fecha trajectória após este silêncio
window_s        = 60     # janela de atribuição (segundos)
window_stride_s = 30     # avanço da janela
inf_cost        = 1e9
new_traj_cost   = 0.5    # custo de abrir trajectória nova (coluna extra)
gender_pen      = 0.4
age_pen         = 0.3
w_temporal      = 0.5
w_spatial       = 0.3
w_attribute     = 0.2


class Stitcher:
    def __init__(self, zones_path):
        with open(zones_path, 'r', encoding='utf-8') as f:
            self.zone_map = json.load(f)['zones']
        self._walk_cache = {}           # cache de Dijkstra já calculado
        self.active_trajectories = {}   # person_id -> list of events
        self.completed_trajectories = []
        self.person_count = 0

    # ── Grafo de zonas ──────────────────────────────────────────────────────

    def get_walk_time(self, from_zone, to_zone):
        """
        Tempo mínimo via Dijkstra — não só adjacências directas.
        Devolve inf se não há caminho.
        Usa cache para não repetir o cálculo.
        """
        if from_zone == to_zone:
            return 0
        key = (from_zone, to_zone)
        if key in self._walk_cache:
            return self._walk_cache[key]

        heap = [(0.0, from_zone)]
        visited = {}
        while heap:
            cost, node = heappop(heap)
            if node in visited:
                continue
            visited[node] = cost
            if node == to_zone:
                self._walk_cache[key] = cost
                return cost
            for neighbour, t in self.zone_map.get(node, {}).get('walk_seconds', {}).items():
                if neighbour not in visited:
                    heappush(heap, (cost + t, neighbour))

        self._walk_cache[key] = float('inf')
        return float('inf')

    # ── Função de custo ─────────────────────────────────────────────────────

    def assignment_cost(self, event, trajectory):
        """
        Custo de associar `event` à `trajectory`.
        Devolve INF_COST se fisicamente impossível.
        """
        last = trajectory[-1]
        gap_s = (event['timestamp'] - last['timestamp']).total_seconds()

        # Restrições hard
        if gap_s < 0:
            return inf_cost
        if gap_s > max_gap_s:
            return inf_cost

        # Zona já ocupada (entry sem exit)?
        zone = event['zone_id']
        entries = sum(1 for e in trajectory if e['zone_id'] == zone and e['event_type'] == 'entry')
        exits   = sum(1 for e in trajectory if e['zone_id'] == zone and e['event_type'] == 'exit')
        if entries > exits:
            return inf_cost

        # Plausibilidade física
        min_walk = self.get_walk_time(last['zone_id'], zone)
        if min_walk != float('inf') and gap_s < min_walk * 0.8:
            return inf_cost

        # Componentes soft
        temporal  = gap_s / max_gap_s

        if min_walk == 0 or min_walk == float('inf'):
            spatial = 0.0
        else:
            ratio   = gap_s / (min_walk * 4)
            spatial = max(0.0, 1.0 - ratio)
            spatial = 1.0 - spatial

        gender_cost = 0.0 if event['gender']    == last['gender']    else gender_pen
        age_cost    = 0.0 if event['age_range'] == last['age_range'] else age_pen
        attribute   = min(gender_cost + age_cost, 1.0)

        return w_temporal * temporal + w_spatial * spatial + w_attribute * attribute

    # ── Gerador de IDs ──────────────────────────────────────────────────────

    def _new_id(self):
        self.person_count += 1
        return f"P_{self.person_count:04d}"

    # ── Loop principal ──────────────────────────────────────────────────────

    def run(self, input_csv, output_csv):
        df = pd.read_csv(input_csv).sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        events_list = df.to_dict('records')
        total = len(events_list)
        print(f"A processar {total:,} eventos")

        # Agrupar eventos por segundo (os eventos do mesmo segundo são um batch)
        from itertools import groupby
        from datetime import timedelta

        # Bucket: truncar timestamp ao segundo
        def to_second(ts):
            return ts.replace(microsecond=0)

        # Agrupar por segundo
        batches = {}
        for e in events_list:
            key = to_second(e['timestamp'])
            if key not in batches:
                batches[key] = []
            batches[key].append(e)

        sorted_keys = sorted(batches.keys())
        processed = 0

        for ts_key in sorted_keys:
            batch = batches[ts_key]

            # Expirar trajectórias inactivas
            inactive = [
                pid for pid, j in self.active_trajectories.items()
                # depois
                if (ts_key - j[-1]['timestamp']).total_seconds() > max_gap_s
            ]
            for pid in inactive:
                self.completed_trajectories.append((pid, self.active_trajectories.pop(pid)))

            # Separar entries de linger/exit
            entries     = [e for e in batch if e['event_type'] == 'entry']
            linger_exit = [e for e in batch if e['event_type'] != 'entry']

            # Associar linger/exit por zona + atributos (greedy simples — não há ambiguidade aqui)
            for ev in linger_exit:
                best_pid, best_gap = None, float('inf')
                for pid, journey in self.active_trajectories.items():
                    last = journey[-1]
                    if last['zone_id'] != ev['zone_id']:
                        continue
                    if last['gender'] != ev['gender'] and last['age_range'] != ev['age_range']:
                        continue
                    gap = (ev['timestamp'] - last['timestamp']).total_seconds()
                    if 0 <= gap < best_gap:
                        best_gap = gap
                        best_pid = pid
                if best_pid:
                    self.active_trajectories[best_pid].append(ev)

            # Hungarian só para entries
            if not entries:
                processed += len(batch)
                continue

            active_list = list(self.active_trajectories.items())
            n_ev = len(entries)
            n_tr = len(active_list)

            if n_tr == 0:
                for ev in entries:
                    pid = self._new_id()
                    self.active_trajectories[pid] = [ev]
            else:
                cost = np.full((n_ev, n_tr + n_ev), inf_cost)
                for i, ev in enumerate(entries):
                    for j, (pid, journey) in enumerate(active_list):
                        cost[i, j] = self.assignment_cost(ev, journey)
                    cost[i, n_tr + i] = new_traj_cost

                row_ind, col_ind = linear_sum_assignment(cost)

                for i, j in zip(row_ind, col_ind):
                    ev = entries[i]
                    if cost[i, j] >= inf_cost or j >= n_tr:
                        pid = self._new_id()
                        self.active_trajectories[pid] = [ev]
                    else:
                        pid, journey = active_list[j]
                        self.active_trajectories[pid].append(ev)

            processed += len(batch)
            if processed % 25000 < len(batch):
                pct = 100 * processed / total
                print(f"  {pct:.0f}% — {processed:,}/{total:,} eventos | "
                    f"trajectórias activas: {len(self.active_trajectories)}", flush=True)

        # Fechar restantes
        for pid, journey in self.active_trajectories.items():
            self.completed_trajectories.append((pid, journey))

        print(f"Concluído. {self.person_count:,} trajectórias reconstruídas.")
        self.save_journeys(output_csv)

    # ── Guardar journeys.csv ────────────────────────────────────────────────

    def save_journeys(self, output_csv):
        """
        Converte trajectórias em linhas do journeys.csv.
        Cada linha = uma visita a uma zona (entry→exit).
        Suporta visitas múltiplas à mesma zona (lista em vez de dict).
        """
        final_data = []
        for pid, events in self.completed_trajectories:
            # Ordenar eventos por tempo
            events = sorted(events, key=lambda e: e['timestamp'])

            # Construir visitas por zona como lista de visitas abertas
            open_visits = {}   # zone_id -> dict da visita actual
            closed_visits = [] # visitas completas

            for e in events:
                z = e['zone_id']
                etype = e['event_type']

                if etype == 'entry':
                    if z in open_visits:
                        # Visita anterior não fechada — fecha-a agora
                        closed_visits.append(open_visits.pop(z))
                    open_visits[z] = {
                        'person_id' : pid,
                        'zone_id'   : z,
                        'entry_time': e['timestamp'],
                        'exit_time' : None,
                        'dwell_s'   : 0,
                        'gender'    : e['gender'],
                        'age_range' : e['age_range'],
                        'visit_date': e['timestamp'].date(),
                        'hour_of_day': e['timestamp'].hour,
                    }
                elif etype == 'linger' and z in open_visits:
                    open_visits[z]['dwell_s'] = e.get('duration_s', 0)

                elif etype == 'exit' and z in open_visits:
                    visit = open_visits.pop(z)
                    visit['exit_time'] = e['timestamp']
                    if visit['dwell_s'] == 0:
                        visit['dwell_s'] = int(
                            (e['timestamp'] - visit['entry_time']).total_seconds()
                        )
                    closed_visits.append(visit)

            # Fechar visitas que não tiveram exit
            for visit in open_visits.values():
                closed_visits.append(visit)

            final_data.extend(closed_visits)

        pd.DataFrame(final_data).to_csv(output_csv, index=False)
        print(f"Guardado: {len(final_data):,} linhas em {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    z_path = "zones.json"
    if not os.path.exists(z_path):
        z_path = os.path.join("..", "zones.json")

    stitcher = Stitcher(z_path)
    stitcher.run(args.input, args.output)