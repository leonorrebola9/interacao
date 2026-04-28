import pandas as pd
import numpy as np
import json
import argparse
import os
from datetime import timedelta
from scipy.optimize import linear_sum_assignment
from heapq import heappush, heappop

# constantes
max_gap_s     = 300
inf_cost      = 1e9
new_traj_cost = 0.25
gender_pen    = 0.4
age_pen       = 0.3
w_temporal    = 0.5
w_spatial     = 0.3
w_attribute   = 0.2

entry_zones    = {'Z_E1', 'Z_E2'}
checkout_zones = {'Z_C1', 'Z_C2', 'Z_C3'}


def load_zone_graph(zones_path):
    with open(zones_path, 'r', encoding='utf-8') as f:
        return json.load(f)['zones']


def get_walk_time(zone_map, cache, from_zone, to_zone):
    if from_zone == to_zone:
        return 0.0
    key = (from_zone, to_zone)
    if key in cache:
        return cache[key]

    heap = [(0.0, from_zone)]
    visited = {}
    while heap:
        cost, node = heappop(heap)
        if node in visited:
            continue
        visited[node] = cost
        if node == to_zone:
            cache[key] = cost
            return cost
        for neighbour, t in zone_map.get(node, {}).get('walk_seconds', {}).items():
            if neighbour not in visited:
                heappush(heap, (cost + t, neighbour))

    cache[key] = float('inf')
    return float('inf')


def assignment_cost(event, traj_events, open_zones, zone_map, cache):
    last  = traj_events[-1]
    gap_s = (event['timestamp'] - last['timestamp']).total_seconds()

    if gap_s < 0:
        return inf_cost
    if gap_s > max_gap_s:
        return inf_cost
    if event['zone_id'] in open_zones:
        return inf_cost

    min_walk = get_walk_time(zone_map, cache, last['zone_id'], event['zone_id'])
    if min_walk != float('inf') and gap_s < min_walk * 0.8:
        return inf_cost

    temporal = gap_s / max_gap_s

    if min_walk == 0 or min_walk == float('inf'):
        spatial = 0.0
    else:
        spatial = 1.0 - max(0.0, 1.0 - gap_s / (min_walk * 4))

    gender_cost = 0.0 if event['gender']    == last['gender']    else gender_pen
    age_cost    = 0.0 if event['age_range'] == last['age_range'] else age_pen
    attribute   = min(gender_cost + age_cost, 1.0)

    return w_temporal * temporal + w_spatial * spatial + w_attribute * attribute


class Stitcher:
    def __init__(self, zones_path):
        self.zone_map  = load_zone_graph(zones_path)
        self.cache     = {}
        self.active    = {}   # pid -> (events, open_zones)
        self.completed = []
        self.count     = 0

    def _new_id(self):
        self.count += 1
        return f"P_{self.count:04d}"

    def _new_traj(self, ev):
        pid        = self._new_id()
        open_zones = {ev['zone_id']} if ev['event_type'] == 'entry' else set()
        self.active[pid] = ([ev], open_zones)

    def _add(self, pid, ev):
        events, open_zones = self.active[pid]
        events.append(ev)
        if ev['event_type'] == 'entry':
            open_zones.add(ev['zone_id'])
        elif ev['event_type'] == 'exit':
            open_zones.discard(ev['zone_id'])

    def run(self, input_csv, output_csv):
        df = pd.read_csv(input_csv).sort_values('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        events_list = df.to_dict('records')
        total = len(events_list)
        print(f"a processar {total:,} eventos")

        # agrupar por segundo
        batches = {}
        for e in events_list:
            key = e['timestamp'].replace(microsecond=0)
            batches.setdefault(key, []).append(e)

        processed = 0
        for ts_key, batch in sorted(batches.items()):

            # expirar trajectórias inactivas
            inactive = [
                pid for pid, (j, _) in self.active.items()
                if (ts_key - j[-1]['timestamp']).total_seconds() > max_gap_s
            ]
            for pid in inactive:
                self.completed.append((pid, self.active.pop(pid)[0]))

            entries     = [e for e in batch if e['event_type'] == 'entry']
            linger_exit = [e for e in batch if e['event_type'] != 'entry']

            # linger/exit: associar à trajectória activa na mesma zona
            for ev in linger_exit:
                best_pid, best_gap = None, float('inf')
                for pid, (journey, _) in self.active.items():
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
                    self._add(best_pid, ev)

            # entradas pela porta: sempre nova trajectória
            door_entries   = [e for e in entries if e['zone_id'] in entry_zones]
            normal_entries = [e for e in entries if e['zone_id'] not in entry_zones]
            for ev in door_entries:
                self._new_traj(ev)

            # Z_CK: associar à trajectória mais recente que passou pelas caixas
            ck_entries     = [e for e in normal_entries if e['zone_id'] == 'Z_CK']
            normal_entries = [e for e in normal_entries if e['zone_id'] != 'Z_CK']
            for ev in ck_entries:
                best_pid, best_gap = None, float('inf')
                for pid, (journey, _) in self.active.items():
                    recent = [e for e in journey
                              if (ev['timestamp'] - e['timestamp']).total_seconds() <= 120]
                    if not any(e['zone_id'] in checkout_zones for e in recent):
                        continue
                    gap = (ev['timestamp'] - journey[-1]['timestamp']).total_seconds()
                    if 0 <= gap < best_gap:
                        best_gap = gap
                        best_pid = pid
                if best_pid:
                    self._add(best_pid, ev)
                else:
                    self._new_traj(ev)

            # hungarian para os restantes entries
            entries = normal_entries
            if not entries:
                processed += len(batch)
                continue

            active_list = list(self.active.items())
            n_ev = len(entries)
            n_tr = len(active_list)

            if n_tr == 0:
                for ev in entries:
                    self._new_traj(ev)
            else:
                cost = np.full((n_ev, n_tr + n_ev), inf_cost)
                for i, ev in enumerate(entries):
                    for j, (pid, (traj_events, open_zones)) in enumerate(active_list):
                        cost[i, j] = assignment_cost(ev, traj_events, open_zones, self.zone_map, self.cache)
                    cost[i, n_tr + i] = new_traj_cost

                row_ind, col_ind = linear_sum_assignment(cost)
                for i, j in zip(row_ind, col_ind):
                    ev = entries[i]
                    if cost[i, j] >= inf_cost or j >= n_tr:
                        self._new_traj(ev)
                    else:
                        self._add(active_list[j][0], ev)

            processed += len(batch)
            if processed % 25000 < len(batch):
                pct = 100 * processed / total
                print(f"  {pct:.0f}% — {processed:,}/{total:,} | activas: {len(self.active)}", flush=True)

        for pid, (journey, _) in self.active.items():
            self.completed.append((pid, journey))

        print(f"concluído: {self.count:,} trajectórias")
        self._save(output_csv)

    def _save(self, output_csv):
        rows = []
        for pid, events in self.completed:
            events      = sorted(events, key=lambda e: e['timestamp'])
            open_visits = {}
            closed      = []

            for e in events:
                z     = e['zone_id']
                etype = e['event_type']

                if etype == 'entry':
                    if z in open_visits:
                        closed.append(open_visits.pop(z))
                    open_visits[z] = {
                        'person_id'  : pid,
                        'zone_id'    : z,
                        'entry_time' : e['timestamp'],
                        'exit_time'  : None,
                        'dwell_s'    : 0,
                        'gender'     : e['gender'],
                        'age_range'  : e['age_range'],
                        'visit_date' : e['timestamp'].date(),
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
                    closed.append(visit)

            for visit in open_visits.values():
                closed.append(visit)
            rows.extend(closed)

        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"guardado: {len(rows):,} linhas em {output_csv}")


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