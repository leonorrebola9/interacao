import argparse
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ─── Parâmetros configuráveis ─────────────────────────────────────────────────

config = {
    "window_s":      30,     # tamanho da janela temporal em segundos
    "overlap_s":     5,      # overlap entre janelas para não partir eventos fronteira
    "max_gap_s":     120,    # 2 minutos sem deteção → fecha trajectória
    "cost_infinite": 1e6,    # custo para associações fisicamente impossíveis
    "max_cost":      500,    # custo máximo aceitável para uma associação válida
}

entry_zones   = {"Z_E1", "Z_E2"}
checkout_exit = {"Z_CK"}
door_zones    = {"Z_E1", "Z_E2"}   # entrada se for a 1ª visita; saída se já iniciada

# ─── Estruturas de dados ──────────────────────────────────────────────────────

@dataclass
class ZoneVisit:
    """Visita completa a uma zona: entry + (opcional linger) + exit."""
    event_id:   str
    zone_id:    str
    entry_time: datetime
    exit_time:  datetime
    dwell_s:    int
    gender:     str
    age_range:  str

@dataclass
class Trajectory:
    """Trajectória em construção de uma pessoa."""
    person_id:      str
    visits:         list = field(default_factory=list)
    last_exit_time: Optional[datetime] = None
    last_zone:      Optional[str] = None
    gender:         str = ""
    age_range:      str = ""
    is_closed:      bool = False
    started:        bool = False   # True após receber a primeira visita

# ─── Grafo de zonas ───────────────────────────────────────────────────────────

def load_zone_graph(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {z: info.get("walk_seconds", {}) for z, info in data["zones"].items()}

def min_travel_time(zone_a: str, zone_b: str, graph: dict) -> int:
    if zone_a == zone_b:
        return 0
    dist = {zone_a: 0}
    heap = [(0, zone_a)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == zone_b:
            return cost
        if cost > dist.get(node, float("inf")):
            continue
        for nb, w in graph.get(node, {}).items():
            nc = cost + w
            if nc < dist.get(nb, float("inf")):
                dist[nb] = nc
                heapq.heappush(heap, (nc, nb))
    return 9999

# ─── Fase 1: construir ZoneVisits ─────────────────────────────────────────────

def build_zone_visits(df: pd.DataFrame) -> list:
    pending = defaultdict(list)
    visits  = []

    for _, row in df.iterrows():
        z     = row["zone_id"]
        t     = row["timestamp"]
        etype = row["event_type"]
        g     = row["gender"]
        age   = row["age_range"]
        dur   = int(row["duration_s"]) if pd.notna(row["duration_s"]) else 0

        if etype == "entry":
            pending[z].append({
                "event_id":    row["event_id"],
                "entry_time":  t,
                "gender":      g,
                "age_range":   age,
                "dwell_s":     0,
                "linger_seen": False,
            })
        elif etype == "linger":
            for p in reversed(pending[z]):
                if not p["linger_seen"] and p["gender"] == g and p["age_range"] == age:
                    p["dwell_s"] = dur
                    p["linger_seen"] = True
                    break
        elif etype == "exit":
            matched = next(
                (p for p in reversed(pending[z]) if p["gender"] == g and p["age_range"] == age),
                None
            )
            if matched:
                pending[z].remove(matched)
                visits.append(ZoneVisit(
                    event_id   = matched["event_id"],
                    zone_id    = z,
                    entry_time = matched["entry_time"],
                    exit_time  = t,
                    dwell_s    = matched["dwell_s"],
                    gender     = matched["gender"],
                    age_range  = matched["age_range"],
                ))

    for z, plist in pending.items():
        for p in plist:
            visits.append(ZoneVisit(
                event_id   = p["event_id"],
                zone_id    = z,
                entry_time = p["entry_time"],
                exit_time  = p["entry_time"] + timedelta(seconds=max(p["dwell_s"], 60)),
                dwell_s    = p["dwell_s"],
                gender     = p["gender"],
                age_range  = p["age_range"],
            ))

    visits.sort(key=lambda v: v.entry_time)
    return visits

# ─── Fase 2: função de custo ──────────────────────────────────────────────────

def trajectory_has_ended(traj: Trajectory) -> bool:
    """Uma trajectória termina quando passa por Z_CK ou por Z_E após já ter começado."""
    if not traj.started:
        return False
    if traj.last_zone in checkout_exit:
        return True
    if traj.last_zone in door_zones:
        return True
    return False

def compute_cost(visit: ZoneVisit, traj: Trajectory, graph: dict) -> float:
    if traj.is_closed:
        return config["cost_infinite"]
    if trajectory_has_ended(traj):
        return config["cost_infinite"]

    if traj.last_exit_time:
        if visit.entry_time < traj.last_exit_time:
            return config["cost_infinite"]
        gap_s = (visit.entry_time - traj.last_exit_time).total_seconds()
        if gap_s > config["max_gap_s"]:
            return config["cost_infinite"]
        if traj.last_zone:
            min_t = min_travel_time(traj.last_zone, visit.zone_id, graph)
            if gap_s < min_t * 0.5:
                return config["cost_infinite"]
            time_cost = gap_s / config["max_gap_s"] * 100
        else:
            time_cost = 0.0
    else:
        time_cost = 0.0

    demo_cost = 0.0
    if traj.gender and traj.gender != visit.gender:
        demo_cost += 200.0
    if traj.age_range and traj.age_range != visit.age_range:
        demo_cost += 120.0

    return time_cost + demo_cost

# ─── Fase 3: Hungarian em janelas temporais ───────────────────────────────────

def add_visit_to_traj(visit: ZoneVisit, traj: Trajectory):
    traj.visits.append(visit)
    traj.last_exit_time = visit.exit_time
    traj.last_zone      = visit.zone_id
    traj.started        = True

def stitch_hungarian(visits: list, graph: dict) -> list:
    if not visits:
        return []

    open_trajs     = []
    closed_trajs   = []
    person_counter = 1

    window  = timedelta(seconds=config["window_s"])
    overlap = timedelta(seconds=config["overlap_s"])

    current_window_start = visits[0].entry_time
    t_end                = visits[-1].entry_time

    while current_window_start <= t_end:
        current_window_end = current_window_start + window
        window_visits = [
            v for v in visits
            if current_window_start <= v.entry_time < current_window_end
        ]

        # Fecha trajectórias terminadas ou com timeout
        still_open = []
        for traj in open_trajs:
            if trajectory_has_ended(traj):
                traj.is_closed = True
                closed_trajs.append(traj)
            elif traj.last_exit_time:
                gap = (current_window_start - traj.last_exit_time).total_seconds()
                if gap > config["max_gap_s"]:
                    traj.is_closed = True
                    closed_trajs.append(traj)
                else:
                    still_open.append(traj)
            else:
                still_open.append(traj)
        open_trajs = still_open

        if not window_visits:
            current_window_start += window - overlap
            continue

        n_v = len(window_visits)
        n_t = len(open_trajs)

        if n_t == 0:
            for v in window_visits:
                traj = Trajectory(f"P_{person_counter:05d}", gender=v.gender, age_range=v.age_range)
                person_counter += 1
                add_visit_to_traj(v, traj)
                open_trajs.append(traj)
            current_window_start += window - overlap
            continue

        size = max(n_v, n_t)
        cost_matrix = np.full((size, size), config["cost_infinite"])

        for i, v in enumerate(window_visits):
            for j, traj in enumerate(open_trajs):
                cost_matrix[i, j] = compute_cost(v, traj, graph)

        for i in range(n_v, size):
            cost_matrix[i, :] = config["max_cost"]
        for j in range(n_t, size):
            cost_matrix[:, j] = config["max_cost"]

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned = set()
        for r, c in zip(row_ind, col_ind):
            if r < n_v and c < n_t and cost_matrix[r, c] < config["max_cost"]:
                add_visit_to_traj(window_visits[r], open_trajs[c])
                assigned.add(r)

        for i, v in enumerate(window_visits):
            if i not in assigned:
                traj = Trajectory(f"P_{person_counter:05d}", gender=v.gender, age_range=v.age_range)
                person_counter += 1
                add_visit_to_traj(v, traj)
                open_trajs.append(traj)

        current_window_start += window - overlap

    for t in open_trajs:
        t.is_closed = True
        closed_trajs.append(t)

    return closed_trajs

# ─── Output ───────────────────────────────────────────────────────────────────

def trajectories_to_df(trajs: list) -> pd.DataFrame:
    rows = []
    for traj in trajs:
        for v in traj.visits:
            rows.append({
                "person_id":   traj.person_id,
                "zone_id":     v.zone_id,
                "entry_time":  v.entry_time,
                "exit_time":   v.exit_time,
                "dwell_s":     v.dwell_s,
                "gender":      traj.gender,
                "age_range":   traj.age_range,
                "visit_date":  v.entry_time.date(),
                "hour_of_day": v.entry_time.hour,
            })
    return pd.DataFrame(rows)

def print_quality_metrics(df: pd.DataFrame):
    print("\n=== Métricas de Qualidade ===")
    total = df["person_id"].nunique()
    print(f"Trajectórias reconstruídas : {total}")
    print(f"Visitas a zonas            : {len(df)}")
    print(f"Média de zonas/trajectória : {len(df)/max(total,1):.1f}")

    violations = 0
    for pid, grp in df.groupby("person_id"):
        grp = grp.sort_values("entry_time")
        for i in range(1, len(grp)):
            if pd.notna(grp.iloc[i-1]["exit_time"]) and \
               grp.iloc[i]["entry_time"] < grp.iloc[i-1]["exit_time"]:
                violations += 1
                break
    print(f"Consistência               : {100*(1-violations/max(total,1)):.1f}%  ({violations} violações)")

    complete = 0
    for pid, grp in df.groupby("person_id"):
        grp = grp.sort_values("entry_time")
        first = grp.iloc[0]["zone_id"]
        last  = grp.iloc[-1]["zone_id"]
        if first in entry_zones and last in (checkout_exit | door_zones):
            complete += 1
    print(f"Completude                 : {100*complete/max(total,1):.1f}%  ({complete}/{total})")
    print(f"  (nota: completude limitada pela cobertura das câmeras nas entradas)")

    gaps = []
    for pid, grp in df.groupby("person_id"):
        grp = grp.sort_values("entry_time")
        for i in range(1, len(grp)):
            prev_exit = grp.iloc[i-1]["exit_time"]
            cur_entry = grp.iloc[i]["entry_time"]
            if pd.notna(prev_exit) and cur_entry >= prev_exit:
                gaps.append((cur_entry - prev_exit).total_seconds())
    if gaps:
        gaps.sort()
        print(f"Gap médio entre zonas      : {sum(gaps)/len(gaps):.1f}s  "
              f"(mediana {gaps[len(gaps)//2]:.1f}s, "
              f"p95 {gaps[int(0.95*len(gaps))]:.1f}s)")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/events.csv")
    parser.add_argument("--output", default="output/journeys.csv")
    parser.add_argument("--zones",  default="data/zones.json")
    args = parser.parse_args()

    print(f"[1/4] A carregar '{args.input}'")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"      {len(df)} eventos.")

    print("[2/4] A carregar mapa de zonas")
    graph = load_zone_graph(args.zones)

    print("[3/4] A construir ZoneVisits")
    visits = build_zone_visits(df)
    print(f"      {len(visits)} visitas a zonas construídas.")

    print("[4/4] A fazer stitching (Hungarian em janelas de 30s)")
    trajs = stitch_hungarian(visits, graph)
    print(f"      {len(trajs)} trajectórias reconstruídas.")

    df_out = trajectories_to_df(trajs)
    df_out.to_csv(args.output, index=False)
    print(f"\nEscrito em '{args.output}'")
    print_quality_metrics(df_out)

if __name__ == "__main__":
    main()