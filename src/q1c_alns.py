"""
Q1c Main Method: ALNS + LAHC for charging station improvement.
 
Adaptive Large Neighbourhood Search with Late Acceptance Hill-Climbing
acceptance criterion (Burke & Bykov, 2017).
 
Destroy operators:
  D1. Random-Remove    — remove k random stations
  D2. Worst-Remove     — remove stations with largest total per-station cost (fixed
                         + ops). Often targets busy hubs, not worst cost/robot; R3
                         repair may still improve layout.
  D3. Zone-Remove      — remove all stations in a geographic zone
  D4. Merge-Stations   — merge two nearest stations (reduces k by 1)
  D5. Split-Station    — split most-overloaded station (increases k by 1)
 
Repair operators:
  R1. Greedy-Insert  — assign unassigned robots to nearest feasible station
  R2. Regret-Insert  — prioritise robots with large regret values
  R3. Cluster-Repair — k-medoids re-clustering of unassigned robots
 
Local operators:
  L1. Station-Relocate — Weiszfeld-like shift towards assigned robots
  L2. Charger-Adjust   — optimise charger counts
  L3. Robot-Swap        — swap robots between neighbouring stations
  L4. Robot-Reassign    — move a single robot to a different station
 
Usage:
    python q1c_alns.py   (from src/), or
    python src/q1c_alns.py   (from project root)

This module is the **sole Q1c** entry point (deterministic improvement after Q1b).
``q2_salns`` and ``pilot_convergence`` import operators / ``alns_lahc`` from here.

    main(): subsets 20/50/100 (30k iters each) + full 1072 (Q1b k=120 stations, 20k iters, LAHC L~4%).
    Outputs: results/q1c_alns_results.json, q1c_alns_output.txt
"""
 
import numpy as np
import time
import json
import sys
import os
import re
from pathlib import Path
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

sys.path.insert(0, ".")

from data_loader import load_data, get_subset
from solution import (
    Solution, CostBreakdown, evaluate, check_feasibility,
    update_transport_flags, update_charger_counts, remove_empty_stations,
    robot_distances,
)
from utils import distance_matrix, haversine
from q1b_construction import construct_solution, kmedoids


def _parse_pounds_token(s: str) -> float:
    return float(s.replace(",", ""))


def parse_q1a_costs_from_output_txt(txt_path: Path) -> dict:
    """Parse Haversine cost per instance from ``q1a_output.txt`` summary table.

    Keys like ``20 robots``, ``50 robots``. Values: Haversine TOTAL (£).
    """
    out = {}
    if not txt_path.is_file():
        return out
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    in_summary = False
    for line in lines:
        if "Q1a MINLP Summary" in line:
            in_summary = True
            continue
        if not in_summary:
            continue
        if line.strip().startswith("-"):
            continue
        m = re.match(r"(\d+)\s+robots\s+", line.strip())
        if not m:
            if out and "Results saved" in line:
                break
            continue
        pounds = re.findall(r"£([\d,]+(?:\.\d+)?)", line)
        if len(pounds) >= 2:
            out[f"{m.group(1)} robots"] = _parse_pounds_token(pounds[-1])
        elif len(pounds) == 1:
            out[f"{m.group(1)} robots"] = _parse_pounds_token(pounds[0])
    return out


def parse_q1b_costs_from_output_txt(txt_path: Path) -> dict:
    """Parse construction TOTAL per instance from ``q1b_output.txt``.

    Keys like ``20 robots`` or ``1072 robots (full)``. Values: TOTAL (£).
    """
    out = {}
    if not txt_path.is_file():
        return out
    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines):
        m = re.search(
            r"Q1b Construction:\s*(\d+)\s+robots(?:\s+\(full\))?",
            line,
        )
        if not m:
            continue
        n = m.group(1)
        label = f"{n} robots (full)" if "(full)" in line else f"{n} robots"
        total = None
        for j in range(i + 1, min(i + 45, len(lines))):
            if re.match(r"Q1b Construction:\s", lines[j]):
                break
            tm = re.search(
                r"TOTAL:\s*£([\d,]+(?:\.\d+)?)",
                lines[j],
                re.IGNORECASE,
            )
            if tm:
                total = _parse_pounds_token(tm.group(1))
                break
        if total is not None:
            out[label] = total
    return out


class Tee:
    """Duplicate stdout to both console and a file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
 
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
 
    def flush(self):
        self.stdout.flush()
        self.file.flush()
 
    def close(self):
        sys.stdout = self.stdout
        self.file.close()
 
 
# ============================================================================
# Fast evaluation
# ============================================================================
 
def fast_evaluate(sol, data):
    """Scalar total cost using assigned robot–station distances only (linear in n)."""
    a = sol.assignments
    n = data.n_robots
    rd = np.zeros(n, dtype=np.float64)
    mask = a >= 0
    if np.any(mask):
        rd[mask] = haversine(
            data.robot_lat[mask], data.robot_lon[mask],
            sol.station_lat[a[mask]], sol.station_lon[a[mask]],
        )
    build = data.cb * sol.n_stations
    maint = data.cm * float(np.sum(sol.n_chargers))
    needs_transport = rd > data.robot_range
    charge = data.cc * float(np.sum(rd[~needs_transport]))  # self-flying only
    transport = data.ch * float(np.sum(needs_transport))
    return build + maint + charge + transport
 
 
# ============================================================================
# Destroy operators
# ============================================================================
# Destroy degree caps as fraction of current station count
DESTROY_RANDOM_MAX_RATIO = 0.20   # D1
DESTROY_WORST_MAX_RATIO  = 0.15   # D2
ZONE_REMOVE_MAX_FRAC     = 0.20   # D3


def destroy_random_remove(sol, data, rng, degree=None):
    """D1: Remove k random stations; unassign their robots."""
    if sol.n_stations < 3:
        return set()
    max_k = max(3, int(np.ceil(sol.n_stations * DESTROY_RANDOM_MAX_RATIO)))
    k = degree or rng.integers(1, max(2, max_k))
    k = min(k, max_k, sol.n_stations - 1)
    remove_ids = rng.choice(sol.n_stations, size=min(k, sol.n_stations - 1),
                            replace=False)
    unassigned = set()
    for j in remove_ids:
        members = np.where(sol.assignments == j)[0]
        unassigned.update(members.tolist())
        sol.assignments[members] = -1
    return unassigned
 
 
def destroy_worst_remove(sol, data, rng, degree=None):
    """D2: Remove stations with largest total per-station cost (not cost/robot)."""
    if sol.n_stations < 3:
        return set()
    max_k = max(3, int(np.ceil(sol.n_stations * DESTROY_WORST_MAX_RATIO)))
    k = degree or rng.integers(1, max(2, max_k))
    k = min(k, max_k, sol.n_stations - 1)

    # Compute per-station cost contribution
    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )
    station_costs = np.zeros(sol.n_stations)
    for j in range(sol.n_stations):
        members = np.where(sol.assignments == j)[0]
        if len(members) == 0:
            continue
        dists_j = dmat[members, j]
        needs_tr = dists_j > data.robot_range[members]
        station_costs[j] = (
            data.cb
            + data.cm * sol.n_chargers[j]
            + data.cc * np.sum(dists_j[~needs_tr])  # self-flying only
            + data.ch * np.sum(needs_tr)
        )
 
    # Remove top-k worst stations
    worst_order = np.argsort(-station_costs)
    remove_ids = worst_order[:min(k, sol.n_stations - 1)]
 
    unassigned = set()
    for j in remove_ids:
        members = np.where(sol.assignments == j)[0]
        unassigned.update(members.tolist())
        sol.assignments[members] = -1
    return unassigned
 
 
def destroy_zone_remove(sol, data, rng, degree=None):
    """D3: Remove all stations in a random geographic zone."""
    if sol.n_stations < 3:
        return set()
 
    # Pick a random station as zone centre
    centre_j = rng.integers(sol.n_stations)
    station_dists = haversine(
        sol.station_lat[centre_j], sol.station_lon[centre_j],
        sol.station_lat, sol.station_lon,
    )
    # Remove stations within a radius (percentile-based)
    radius_pct = rng.uniform(0.08, 0.30)
    threshold = np.quantile(station_dists[station_dists > 0],
                            radius_pct) if sol.n_stations > 2 else np.inf
    remove_mask = (station_dists <= threshold) & (station_dists >= 0)
    remove_mask[centre_j] = True

    # Keep at least 1 station
    remove_ids = np.where(remove_mask)[0]
    if len(remove_ids) >= sol.n_stations:
        remove_ids = remove_ids[:sol.n_stations - 1]

    cap = max(2, int(np.ceil(sol.n_stations * ZONE_REMOVE_MAX_FRAC)))
    if len(remove_ids) > cap:
        dist_sub = station_dists[remove_ids]
        nearest_first = np.argsort(dist_sub)
        remove_ids = remove_ids[nearest_first[:cap]]

    unassigned = set()
    for j in remove_ids:
        members = np.where(sol.assignments == j)[0]
        unassigned.update(members.tolist())
        sol.assignments[members] = -1
    return unassigned
 
 
def destroy_merge_stations(sol, data, rng, degree=None):
    """D4: Merge nearest station pairs — reduces k by 1~n_merge.

    Finds the closest pairs of stations and unassigns all robots from the
    smaller ones.  The Repair operator then redistributes them, potentially
    into the surviving neighbours, effectively merging clusters.
    """
    if sol.n_stations < 3:
        return set()

    # Number of merges: 1~3 for variety
    n_merge = degree or rng.integers(1, min(4, sol.n_stations // 2 + 1))

    # Pairwise station distances
    s_dists = distance_matrix(
        sol.station_lat, sol.station_lon,
        sol.station_lat, sol.station_lon,
    )
    np.fill_diagonal(s_dists, np.inf)

    unassigned = set()
    removed = set()

    for _ in range(n_merge):
        if sol.n_stations - len(removed) < 3:
            break

        # Mask already-removed stations
        for r in removed:
            s_dists[r, :] = np.inf
            s_dists[:, r] = np.inf

        j1, j2 = np.unravel_index(np.argmin(s_dists), s_dists.shape)
        if s_dists[j1, j2] == np.inf:
            break

        # Remove the station with fewer robots
        count1 = int(np.sum(sol.assignments == j1))
        count2 = int(np.sum(sol.assignments == j2))
        remove_j = j2 if count1 >= count2 else j1

        members = np.where(sol.assignments == remove_j)[0]
        unassigned.update(members.tolist())
        sol.assignments[members] = -1
        removed.add(remove_j)

    return unassigned
 
 
def destroy_split_station(sol, data, rng, degree=None):
    """D5: Split a high-cost station — increases k by 1.

    Scores stations by total operational cost (transport + charging),
    randomly picks from top-5, then unassigns the farthest half of robots.
    The Repair operator (especially R3 Cluster-Repair) will open a new
    station closer to them, effectively splitting the cluster.
    """
    if sol.n_stations < 2:
        return set()

    # Score each station by total operational cost
    station_scores = np.zeros(sol.n_stations)
    for j in range(sol.n_stations):
        members = np.where(sol.assignments == j)[0]
        if len(members) < 3:
            continue
        dists = haversine(
            data.robot_lat[members], data.robot_lon[members],
            sol.station_lat[j], sol.station_lon[j],
        )
        needs_tr = dists > data.robot_range[members]
        station_scores[j] = (
            data.ch * np.sum(needs_tr)
            + data.cc * np.sum(dists[~needs_tr])
        )

    # Pick randomly from top-5 scorers (adds diversity)
    top_k = min(5, int(np.sum(station_scores > 0)))
    if top_k == 0:
        return set()
    candidates = np.argsort(-station_scores)[:top_k]
    j_big = int(rng.choice(candidates))

    members = np.where(sol.assignments == j_big)[0]
    if len(members) < 3:
        return set()

    # Unassign the farthest half of members
    dists = haversine(
        data.robot_lat[members], data.robot_lon[members],
        sol.station_lat[j_big], sol.station_lon[j_big],
    )
    half = max(1, len(members) // 2)
    far_local = np.argsort(dists)[half:]
    far_global = members[far_local]

    unassigned = set(far_global.tolist())
    sol.assignments[far_global] = -1
    return unassigned
 
 
# ============================================================================
# Repair — optional new station if nearest existing site is far vs range
# ============================================================================
# If min dist to an open site > ratio * range, open a new site (explores multi-site).
REPAIR_OPEN_STATION_DIST_RATIO = 0.8


# ============================================================================
# Repair operators
# ============================================================================

def repair_greedy_insert(sol, data, rng, unassigned):
    """R1: Assign each unassigned robot to the nearest feasible station."""
    if not unassigned:
        return
    unassigned_list = list(unassigned)
 
    # Remove empty stations first
    _compact_stations(sol)
 
    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )
    capacity = np.array([sol.n_chargers[j] * data.q
                         for j in range(sol.n_stations)], dtype=int)
    # Count of robots per station
    counts = np.zeros(sol.n_stations, dtype=int)
    for j in range(sol.n_stations):
        members = np.where(sol.assignments == j)[0]
        counts[j] = len(members)
 
    for i in sorted(unassigned_list, key=lambda x: np.min(dmat[x])):
        min_d = float(np.min(dmat[i]))
        r_i = max(float(data.robot_range[i]), 1e-6)
        if (
            min_d > REPAIR_OPEN_STATION_DIST_RATIO * r_i
            and sol.n_stations < data.n_robots
        ):
            new_j = sol.n_stations
            sol.station_lon = np.append(sol.station_lon, data.robot_lon[i])
            sol.station_lat = np.append(sol.station_lat, data.robot_lat[i])
            sol.n_chargers = np.append(sol.n_chargers, 1)
            capacity = np.append(capacity, int(data.q))
            counts = np.append(counts, 1)
            sol.assignments[i] = new_j
            dmat = distance_matrix(
                data.robot_lat, data.robot_lon,
                sol.station_lat, sol.station_lon,
            )
            continue

        assigned = False
        for j in np.argsort(dmat[i]):
            if counts[j] < capacity[j]:
                sol.assignments[i] = j
                counts[j] += 1
                assigned = True
                break
            elif sol.n_chargers[j] < data.m:
                sol.n_chargers[j] += 1
                capacity[j] = sol.n_chargers[j] * data.q
                sol.assignments[i] = j
                counts[j] += 1
                assigned = True
                break
 
        if not assigned:
            # Open new station at robot's location
            new_j = sol.n_stations
            sol.station_lon = np.append(sol.station_lon, data.robot_lon[i])
            sol.station_lat = np.append(sol.station_lat, data.robot_lat[i])
            sol.n_chargers = np.append(sol.n_chargers, 1)
            capacity = np.append(capacity, int(data.q))
            counts = np.append(counts, 1)
            sol.assignments[i] = new_j
 
    update_transport_flags(sol, data)
 
 
def repair_regret_insert(sol, data, rng, unassigned):
    """R2: Regret-based insertion — prioritise robots with high regret."""
    if not unassigned:
        return
    unassigned_list = list(unassigned)
 
    _compact_stations(sol)
 
    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )
    capacity = np.array([sol.n_chargers[j] * data.q
                         for j in range(sol.n_stations)], dtype=int)
    # Count of robots per station
    counts = np.zeros(sol.n_stations, dtype=int)
    for j in range(sol.n_stations):
        members = np.where(sol.assignments == j)[0]
        counts[j] = len(members)
 
    remaining = set(unassigned_list)
 
    while remaining:
        # Compute regret-2 for each unassigned robot
        best_regret = -np.inf
        best_robot = None
        best_station = None
 
        for i in remaining:
            sorted_js = np.argsort(dmat[i])
            costs = []
            for j in sorted_js:
                if (counts[j] < capacity[j]
                        or sol.n_chargers[j] < data.m):
                    cost = dmat[i, j]
                    costs.append((cost, j))
                    if len(costs) >= 2:
                        break
 
            if len(costs) == 0:
                if best_robot is None:
                    best_robot = i
                    best_station = -1
                continue
 
            regret = costs[1][0] - costs[0][0] if len(costs) >= 2 else costs[0][0]
            if regret > best_regret:
                best_regret = regret
                best_robot = i
                best_station = costs[0][1]
 
        if best_robot is None:
            break
 
        if best_station == -1 or best_station is None:
            # Open new station
            new_j = sol.n_stations
            sol.station_lon = np.append(sol.station_lon,
                                        data.robot_lon[best_robot])
            sol.station_lat = np.append(sol.station_lat,
                                        data.robot_lat[best_robot])
            sol.n_chargers = np.append(sol.n_chargers, 1)
            capacity = np.append(capacity, int(data.q))
            counts = np.append(counts, 1)
            dmat_new = distance_matrix(
                data.robot_lat, data.robot_lon,
                sol.station_lat[new_j:new_j+1],
                sol.station_lon[new_j:new_j+1],
            )
            dmat = np.hstack([dmat, dmat_new])
            sol.assignments[best_robot] = new_j
        else:
            j = best_station
            r_br = max(float(data.robot_range[best_robot]), 1e-6)
            if (
                dmat[best_robot, j] > REPAIR_OPEN_STATION_DIST_RATIO * r_br
                and sol.n_stations < data.n_robots
            ):
                new_j = sol.n_stations
                sol.station_lon = np.append(
                    sol.station_lon, data.robot_lon[best_robot])
                sol.station_lat = np.append(
                    sol.station_lat, data.robot_lat[best_robot])
                sol.n_chargers = np.append(sol.n_chargers, 1)
                capacity = np.append(capacity, int(data.q))
                counts = np.append(counts, 1)
                dmat_new = distance_matrix(
                    data.robot_lat, data.robot_lon,
                    sol.station_lat[new_j:new_j + 1],
                    sol.station_lon[new_j:new_j + 1],
                )
                dmat = np.hstack([dmat, dmat_new])
                sol.assignments[best_robot] = new_j
            else:
                if counts[j] >= capacity[j] \
                        and sol.n_chargers[j] < data.m:
                    sol.n_chargers[j] += 1
                    capacity[j] = sol.n_chargers[j] * data.q
                sol.assignments[best_robot] = j
                counts[j] += 1

        remaining.discard(best_robot)
 
    update_transport_flags(sol, data)
 
 
def repair_cluster(sol, data, rng, unassigned):
    """R3: K-medoids re-clustering of unassigned robots into new stations."""
    if not unassigned:
        return
    unassigned_list = list(unassigned)
 
    _compact_stations(sol)
 
    u_idx = np.array(unassigned_list)
    cap = data.m * data.q
    k_new = max(1, int(np.ceil(len(unassigned_list) / cap)))
 
    # Compute distance matrix among unassigned robots
    u_dm = distance_matrix(
        data.robot_lat[u_idx], data.robot_lon[u_idx],
        data.robot_lat[u_idx], data.robot_lon[u_idx],
    )
 
    medoids, labels = kmedoids(u_dm, k_new, rng=rng)
 
    for c in range(k_new):
        members_local = np.where(labels == c)[0]
        members_global = u_idx[members_local]
        med_global = u_idx[medoids[c]]
 
        new_j = sol.n_stations
        sol.station_lon = np.append(sol.station_lon, data.robot_lon[med_global])
        sol.station_lat = np.append(sol.station_lat, data.robot_lat[med_global])
        cluster_count = len(members_global)
        n_chargers_needed = min(int(np.ceil(cluster_count / data.q)),
                                data.m)
        sol.n_chargers = np.append(sol.n_chargers, n_chargers_needed)
 
        for i in members_global:
            sol.assignments[i] = new_j
 
    # Handle overflow: if a new station exceeds capacity
    _fix_overflow(sol, data)
    update_transport_flags(sol, data)
 
 
def _compact_stations(sol):
    """Remove stations with no assigned robots and reindex."""
    counts = np.bincount(sol.assignments[sol.assignments >= 0],
                         minlength=sol.n_stations)
    keep = counts > 0
    if np.all(keep):
        return
 
    new_idx = np.full(sol.n_stations, -1, dtype=int)
    new_idx[keep] = np.arange(int(np.sum(keep)))
 
    sol.station_lon = sol.station_lon[keep]
    sol.station_lat = sol.station_lat[keep]
    sol.n_chargers = sol.n_chargers[keep]
 
    valid = sol.assignments >= 0
    sol.assignments[valid] = new_idx[sol.assignments[valid]]
 
 
def _fix_overflow(sol, data):
    """Move robots from stations exceeding capacity."""
    cap = data.m * data.q
 
    for j in range(sol.n_stations):
        members = np.where(sol.assignments == j)[0]
        count_j = len(members)
        if count_j <= cap:
            continue
        # Keep first cap robots (nearest by distance), spill the rest
        dists = haversine(
            data.robot_lat[members], data.robot_lon[members],
            sol.station_lat[j], sol.station_lon[j],
        )
        sorted_local = np.argsort(dists)
        keep_count = cap
        spill_local = sorted_local[keep_count:]
        spill_global = members[spill_local]
 
        for i in spill_global:
            # Find nearest station with capacity
            s_dists = haversine(
                data.robot_lat[i], data.robot_lon[i],
                sol.station_lat, sol.station_lon,
            )
            for jj in np.argsort(s_dists):
                if jj == j:
                    continue
                mem_jj = np.where(sol.assignments == jj)[0]
                count_jj = len(mem_jj)
                if count_jj < sol.n_chargers[jj] * data.q:
                    sol.assignments[i] = jj
                    break
                elif sol.n_chargers[jj] < data.m:
                    sol.n_chargers[jj] += 1
                    sol.assignments[i] = jj
                    break
            else:
                # Open new station at robot location
                new_j = sol.n_stations
                sol.station_lon = np.append(sol.station_lon, data.robot_lon[i])
                sol.station_lat = np.append(sol.station_lat, data.robot_lat[i])
                sol.n_chargers = np.append(sol.n_chargers, 1)
                sol.assignments[i] = new_j
 
    update_charger_counts(sol, data)
 
 
# ============================================================================
# Local search operators
# ============================================================================
 
def local_station_relocate(sol, data, rng):
    """L1: Weiszfeld-like step — move station towards its assigned robots."""
    j = rng.integers(sol.n_stations)
    members = np.where(sol.assignments == j)[0]
    if len(members) < 2:
        return False
 
    # Weighted average (inverse-distance weights for Weiszfeld)
    dists = haversine(
        data.robot_lat[members], data.robot_lon[members],
        sol.station_lat[j], sol.station_lon[j],
    )
    weights = 1.0 / (dists + 1e-6)
    weights /= weights.sum()
 
    new_lon = np.average(data.robot_lon[members], weights=weights)
    new_lat = np.average(data.robot_lat[members], weights=weights)
 
    # Damped step
    alpha = rng.uniform(0.3, 0.8)
    sol.station_lon[j] += alpha * (new_lon - sol.station_lon[j])
    sol.station_lat[j] += alpha * (new_lat - sol.station_lat[j])
 
    # Update transport flags
    for i in members:
        d = haversine(
            data.robot_lat[i], data.robot_lon[i],
            sol.station_lat[j], sol.station_lon[j],
        )
        sol.needs_transport[i] = d > data.robot_range[i]
 
    return True
 
 
def local_charger_adjust(sol, data, rng):
    """L2: Try adding or removing a charger at a random station."""
    j = rng.integers(sol.n_stations)
    members = np.where(sol.assignments == j)[0]
    if len(members) == 0:
        return False
    count = len(members)
    needed = int(np.ceil(count / data.q))
 
    if sol.n_chargers[j] > needed and sol.n_chargers[j] > 1:
        sol.n_chargers[j] -= 1
        return True
    elif sol.n_chargers[j] < data.m and count > 0:
        sol.n_chargers[j] += 1
        return True
 
    return False
 
 
def local_robot_swap(sol, data, rng):
    """L3: Swap two robots between different stations."""
    n = data.n_robots
    i1 = rng.integers(n)
    i2 = rng.integers(n)
    if sol.assignments[i1] == sol.assignments[i2]:
        return False

    sol.assignments[i1], sol.assignments[i2] = (
        sol.assignments[i2], sol.assignments[i1]
    )

    for i in [i1, i2]:
        j = sol.assignments[i]
        d = haversine(
            data.robot_lat[i], data.robot_lon[i],
            sol.station_lat[j], sol.station_lon[j],
        )
        sol.needs_transport[i] = d > data.robot_range[i]

    return True
 
 
def local_robot_reassign(sol, data, rng):
    """L4: Move a random robot to another station (if capacity allows)."""
    n = data.n_robots
    i = rng.integers(n)
    old_j = sol.assignments[i]

    if sol.n_stations < 2:
        return False

    new_j = rng.integers(sol.n_stations - 1)
    if new_j >= old_j:
        new_j += 1

    # Check capacity at target station
    count_new = int(np.sum(sol.assignments == new_j))
    cap_new = sol.n_chargers[new_j] * data.q
    if count_new >= cap_new:
        if sol.n_chargers[new_j] < data.m:
            sol.n_chargers[new_j] += 1
        else:
            return False

    sol.assignments[i] = new_j

    # Update old station charger count
    count_old = int(np.sum(sol.assignments == old_j))
    needed_old = max(1, int(np.ceil(count_old / data.q)))
    sol.n_chargers[old_j] = min(needed_old, data.m)

    # Update transport flag
    d = haversine(
        data.robot_lat[i], data.robot_lon[i],
        sol.station_lat[new_j], sol.station_lon[new_j],
    )
    sol.needs_transport[i] = d > data.robot_range[i]

    return True


# ============================================================================
# ALNS + LAHC
# ============================================================================

DESTROY_OPS = [
    ("D: Random-Remove",  destroy_random_remove),
    ("D: Worst-Remove",   destroy_worst_remove),
    ("D: Zone-Remove",    destroy_zone_remove),
    ("D: Merge-Stations", destroy_merge_stations),  # reduces k
    ("D: Split-Station",  destroy_split_station),   # increases k
]
 
REPAIR_OPS = [
    ("R: Greedy-Insert", repair_greedy_insert),
    ("R: Regret-Insert", repair_regret_insert),
    ("R: Cluster-Repair", repair_cluster),
]
 
LOCAL_OPS = [
    ("L: Station-Relocate", local_station_relocate),
    ("L: Charger-Adjust", local_charger_adjust),
    ("L: Robot-Swap", local_robot_swap),
    ("L: Robot-Reassign", local_robot_reassign),
]
 
 
def alns_lahc(sol_init, data, max_iter=50000, L=5000,
              segment_length=200, reaction_factor=0.1,
              sigma1=33, sigma2=9, sigma3=1,
              p_large=0.60, rng=None, verbose=True):
    """ALNS with LAHC acceptance criterion.
 
    Args:
        sol_init:        Initial solution.
        data:            Problem data.
        max_iter:        Maximum iterations.
        L:               LAHC history length.
        segment_length:  Iterations per weight-update segment.
        reaction_factor: Controls how quickly weights adapt (rho).
        sigma1/2/3:      Rewards for new best / improving / accepted.
        p_large:         Prob. of destroy+repair (lower → more local moves).
        rng:             Random generator.
        verbose:         Print progress (~10 lines per run: every max_iter//10
                         iterations; default max_iter=50000 → every 5000).
 
    Returns:
        best_sol, best_cost, history, station_count_history, weight_history, stats
    """
    if rng is None:
        rng = np.random.default_rng(42)
 
    # Operator weights — initialise uniformly
    n_d = len(DESTROY_OPS)
    n_r = len(REPAIR_OPS)
    n_l = len(LOCAL_OPS)
 
    d_weights = np.ones(n_d)
    r_weights = np.ones(n_r)
    l_weights = np.ones(n_l)
 
    d_scores = np.zeros(n_d)
    r_scores = np.zeros(n_r)
    l_scores = np.zeros(n_l)
 
    d_uses = np.zeros(n_d, dtype=int)
    r_uses = np.zeros(n_r, dtype=int)
    l_uses = np.zeros(n_l, dtype=int)
 
    # Track total stats
    d_total_uses = np.zeros(n_d, dtype=int)
    r_total_uses = np.zeros(n_r, dtype=int)
    l_total_uses = np.zeros(n_l, dtype=int)
    d_total_improves = np.zeros(n_d, dtype=int)
    r_total_improves = np.zeros(n_r, dtype=int)
    l_total_improves = np.zeros(n_l, dtype=int)
 
    # Weight history for plotting
    n_segments = max_iter // segment_length + 1
    d_weight_hist = np.zeros((n_segments, n_d))
    r_weight_hist = np.zeros((n_segments, n_r))
    l_weight_hist = np.zeros((n_segments, n_l))
    seg_idx = 0
 
    # Initialise
    current = sol_init.copy()
    current_cost = fast_evaluate(current, data)
    best = current.copy()
    best_cost = current_cost
 
    # LAHC history
    cost_history = np.full(L, current_cost)
    history = np.zeros(max_iter)
    station_count_history = np.zeros(max_iter, dtype=np.float64)

    t0 = time.time()
    # ~10 progress lines per run; avoid % 0 when max_iter < 10
    progress_interval = max(max_iter // 10, 1)

    for it in range(max_iter):
        # Choose: large neighbourhood (destroy+repair) or local move
        if rng.random() < p_large:
            # Select destroy and repair operators (roulette wheel)
            d_probs = d_weights / d_weights.sum()
            r_probs = r_weights / r_weights.sum()
            d_idx = rng.choice(n_d, p=d_probs)
            r_idx = rng.choice(n_r, p=r_probs)
 
            d_uses[d_idx] += 1
            r_uses[r_idx] += 1
            d_total_uses[d_idx] += 1
            r_total_uses[r_idx] += 1
 
            candidate = current.copy()
            unassigned = DESTROY_OPS[d_idx][1](candidate, data, rng)
            REPAIR_OPS[r_idx][1](candidate, data, rng, unassigned)
 
            # Ensure feasibility after repair
            _compact_stations(candidate)
            update_charger_counts(candidate, data)
 
            cand_cost = fast_evaluate(candidate, data)
 
            # LAHC acceptance
            lahc_ref = cost_history[it % L]
            old_cost = current_cost  # save BEFORE update for scoring
            if cand_cost < lahc_ref or cand_cost < current_cost:
                current = candidate
                current_cost = cand_cost
                cost_history[it % L] = current_cost
 
                if cand_cost < best_cost - 1e-6:
                    best = current.copy()
                    best_cost = cand_cost
                    d_scores[d_idx] += sigma1
                    r_scores[r_idx] += sigma1
                    d_total_improves[d_idx] += 1
                    r_total_improves[r_idx] += 1
                elif cand_cost < old_cost - 1e-6:
                    d_scores[d_idx] += sigma2
                    r_scores[r_idx] += sigma2
                else:
                    d_scores[d_idx] += sigma3
                    r_scores[r_idx] += sigma3
            else:
                cost_history[it % L] = current_cost
 
        else:
            # Local operator
            l_probs = l_weights / l_weights.sum()
            l_idx = rng.choice(n_l, p=l_probs)
            l_uses[l_idx] += 1
            l_total_uses[l_idx] += 1
 
            candidate = current.copy()
            success = LOCAL_OPS[l_idx][1](candidate, data, rng)
 
            if not success:
                history[it] = best_cost
                station_count_history[it] = float(current.n_stations)
                continue
 
            cand_cost = fast_evaluate(candidate, data)
 
            lahc_ref = cost_history[it % L]
            old_cost = current_cost  # save BEFORE update for scoring
            if cand_cost < lahc_ref or cand_cost < current_cost:
                current = candidate
                current_cost = cand_cost
                cost_history[it % L] = current_cost
 
                if cand_cost < best_cost - 1e-6:
                    best = current.copy()
                    best_cost = cand_cost
                    l_scores[l_idx] += sigma1
                    l_total_improves[l_idx] += 1
                elif cand_cost < old_cost - 1e-6:
                    l_scores[l_idx] += sigma2
                else:
                    l_scores[l_idx] += sigma3
            else:
                cost_history[it % L] = current_cost
 
        history[it] = best_cost
        station_count_history[it] = float(current.n_stations)

        # Weight update at end of each segment
        if (it + 1) % segment_length == 0:
            rho = reaction_factor
            for k in range(n_d):
                if d_uses[k] > 0:
                    d_weights[k] = (1 - rho) * d_weights[k] + \
                                   rho * d_scores[k] / d_uses[k]
                d_weights[k] = max(d_weights[k], 0.05)
            for k in range(n_r):
                if r_uses[k] > 0:
                    r_weights[k] = (1 - rho) * r_weights[k] + \
                                   rho * r_scores[k] / r_uses[k]
                r_weights[k] = max(r_weights[k], 0.05)
            for k in range(n_l):
                if l_uses[k] > 0:
                    l_weights[k] = (1 - rho) * l_weights[k] + \
                                   rho * l_scores[k] / l_uses[k]
                l_weights[k] = max(l_weights[k], 0.05)
 
            if seg_idx < n_segments:
                d_weight_hist[seg_idx] = d_weights.copy()
                r_weight_hist[seg_idx] = r_weights.copy()
                l_weight_hist[seg_idx] = l_weights.copy()
                seg_idx += 1
 
            # Reset segment counters
            d_scores[:] = 0
            r_scores[:] = 0
            l_scores[:] = 0
            d_uses[:] = 0
            r_uses[:] = 0
            l_uses[:] = 0
 
        if verbose and (it + 1) % progress_interval == 0:
            elapsed = time.time() - t0
            print(f"  ALNS iter {it+1:>6}/{max_iter}: "
                  f"best=£{best_cost:,.0f}  curr=£{current_cost:,.0f}  "
                  f"time={elapsed:.1f}s")
 
    elapsed = time.time() - t0
 
    # Clean up best solution
    remove_empty_stations(best)
    update_charger_counts(best, data)
    update_transport_flags(best, data)
 
    # Build weight history dict for plotting
    weight_history = {}
    actual_segs = seg_idx
    for k in range(n_d):
        weight_history[DESTROY_OPS[k][0]] = d_weight_hist[:actual_segs, k]
    for k in range(n_r):
        weight_history[REPAIR_OPS[k][0]] = r_weight_hist[:actual_segs, k]
    for k in range(n_l):
        weight_history[LOCAL_OPS[k][0]] = l_weight_hist[:actual_segs, k]
 
    stats = {
        "time": elapsed,
        "iterations": max_iter,
        "L": L,
        "segment_length": segment_length,
        "operator_stats": {},
    }
 
    all_ops = (
        [(DESTROY_OPS[k][0], d_total_uses[k], d_total_improves[k])
         for k in range(n_d)] +
        [(REPAIR_OPS[k][0], r_total_uses[k], r_total_improves[k])
         for k in range(n_r)] +
        [(LOCAL_OPS[k][0], l_total_uses[k], l_total_improves[k])
         for k in range(n_l)]
    )
    for name, uses, improves in all_ops:
        stats["operator_stats"][name] = {
            "uses": int(uses),
            "improves": int(improves),
        }
 
    return best, best_cost, history, station_count_history, weight_history, stats
 
 
# ============================================================================
# Solve wrapper
# ============================================================================
 
def solve_instance(data, label="", max_iter=50000, rng=None, k=None, L=5000):
    """Build initial solution, then improve with ALNS+LAHC."""
    print(f"\n{'=' * 60}")
    print(f"Q1c ALNS+LAHC: {label}")
    print(f"  n={data.n_robots}, max_iter={max_iter}"
          + (f", k={k}" if k else ""))
    print(f"{'=' * 60}")
 
    # 1. Construct initial solution
    rng_construct = np.random.default_rng(42)
    sol_init = construct_solution(data, k=k, rng=rng_construct)
    init_cost = evaluate(sol_init, data)
    print(f"  Initial (Q1b): £{init_cost.total:,.2f}  "
          f"({sol_init.n_stations} stations)")
 
    # 2. Run ALNS+LAHC
    if rng is None:
        rng = np.random.default_rng(456)
 
    best, best_cost, history, station_count_history, weight_history, stats = alns_lahc(
        sol_init, data, max_iter=max_iter, L=L, rng=rng,
    )
 
    cost = evaluate(best, data)
    feasible, violations = check_feasibility(best, data)
    improvement = (init_cost.total - cost.total) / init_cost.total * 100
 
    print(f"\n  ALNS+LAHC Result:")
    print(f"    Time:        {stats['time']:.3f}s")
    print(f"    Stations:    {best.n_stations}")
    print(f"    Chargers:    {best.n_chargers.tolist()}")
    print(f"    Transport:   {int(np.sum(best.needs_transport))} robots")
    print(f"    Cost breakdown:")
    print(f"      Build:       £{cost.build:,.2f}")
    print(f"      Maintenance: £{cost.maintenance:,.2f}")
    print(f"      Charging:    £{cost.charging:,.2f}")
    print(f"      Transport:   £{cost.transport:,.2f}")
    print(f"      TOTAL:       £{cost.total:,.2f}")
    print(f"    Improvement: {improvement:.2f}%")
    print(f"    Feasible:    {feasible}")
    if not feasible:
        for v in violations:
            print(f"      - {v}")
 
    print(f"\n  Operator performance:")
    for name, s in stats["operator_stats"].items():
        if s["uses"] > 0:
            print(f"    {name:<25} uses={s['uses']:>6}  "
                  f"improves={s['improves']}")
 
    return {
        "sol_init": sol_init,
        "sol": best,
        "init_cost": init_cost,
        "cost": cost,
        "history": history,
        "station_count_history": station_count_history,
        "weight_history": weight_history,
        "stats": stats,
        "feasible": feasible,
        "improvement": improvement,
    }
 
 
# ============================================================================
# Main
# ============================================================================
 
def main():
    """Run small subsets (30k iters) and full instance (20k iters, Q1b k=120).

    k=120 matches Q1b k-scan best station count. LAHC L ~ 4% of max_iter.
    """
    FULL_K = 120
    FULL_MAX_ITER = 20_000
    FULL_LAHC_L = 800  # ≈ 4% * FULL_MAX_ITER
    SMALL_MAX_ITER = 30_000

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_RESULTS_DIR / "q1c_alns_output.txt")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    tee = Tee(log_path)
    sys.stdout = tee

    print("Q1c ALNS+LAHC — small instances + full (k=%d, %d iters)"
          % (FULL_K, FULL_MAX_ITER))
    print("=" * 60)

    data = load_data()
    print(f"Full data loaded: {data.n_robots} robots\n")

    rng = np.random.default_rng(456)

    # Small instances: Q1b default k, 30k ALNS iters
    small_sizes = [20, 50, 100]
    rng_subset = np.random.default_rng(42)
    small_results = {}

    for n_robots in small_sizes:
        idx = rng_subset.choice(data.n_robots, n_robots, replace=False)
        sub = get_subset(data, idx)
        label = f"{n_robots} robots"

        r = solve_instance(sub, label=label, max_iter=SMALL_MAX_ITER, rng=rng)
        small_results[label] = r

    # Full: pass FULL_LAHC_L; small runs keep solve_instance default L=5000
    full_result = solve_instance(
        data,
        label="1072 robots (full)",
        max_iter=FULL_MAX_ITER,
        rng=rng,
        k=FULL_K,
        L=FULL_LAHC_L,
    )
 
    # ── Visualisations ─────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization import (
        plot_solution_map, plot_solution_comparison,
        plot_cost_breakdown, plot_cost_comparison,
        plot_convergence, plot_operator_weights,
    )
 
    # Fig-C2: ALNS solution map
    fig, _ = plot_solution_map(
        full_result["sol"], data,
        title="Q1c ALNS+LAHC — 1072 robots (Fig-C2)",
        save_path=str(_RESULTS_DIR / "q1c_alns_map_1072.png"))
    plt.close(fig)
 
    # Fig-C3: Before/After
    fig = plot_solution_comparison(
        full_result["sol_init"], full_result["sol"], data,
        titles=("Q1b Construction", "Q1c ALNS+LAHC"),
        save_path=str(_RESULTS_DIR / "q1c_alns_comparison_1072.png"))
    plt.close(fig)
 
    # Cost breakdown (ALNS)
    fig, _ = plot_cost_breakdown(
        full_result["cost"],
        title="Q1c ALNS+LAHC Cost Breakdown — 1072 robots",
        save_path=str(_RESULTS_DIR / "q1c_alns_cost_1072.png"))
    plt.close(fig)
 
    # Fig-C5: Convergence (ALNS+LAHC only; SA baseline removed)
    fig, _ = plot_convergence(
        {"ALNS": full_result["history"]},
        title="Q1c Convergence — 1072 robots (Fig-C5)",
        save_path=str(_RESULTS_DIR / "q1c_convergence_1072.png"))
    plt.close(fig)
 
    # Fig-C6: Operator weights
    fig, _ = plot_operator_weights(
        full_result["weight_history"],
        title="Q1c ALNS Operator Weights — 1072 robots (Fig-C6)",
        save_path=str(_RESULTS_DIR / "q1c_operator_weights_1072.png"))
    plt.close(fig)

    # Extra plots: station count trace and best-cost trace (1072)
    k_hist = full_result["station_count_history"]
    cost_hist = full_result["history"]
    iters = np.arange(len(cost_hist))

    fig_k, ax_k = plt.subplots(figsize=(12, 4))
    ax_k.plot(iters, k_hist, linewidth=0.5, color="#3498DB", alpha=0.85)
    ax_k.set_xlabel("Iteration")
    ax_k.set_ylabel("Number of stations (current solution)")
    ax_k.set_title(
        f"Q1c ALNS+LAHC — Station count vs iteration (1072 robots, Q1b k={FULL_K})"
    )
    ax_k.grid(True, alpha=0.3)
    ax_k.axhline(
        y=full_result["sol"].n_stations,
        color="#E74C3C",
        linestyle="--",
        linewidth=1,
        label=f"Final k = {full_result['sol'].n_stations}",
    )
    ax_k.axhline(
        y=full_result["sol_init"].n_stations,
        color="#95A5A6",
        linestyle="--",
        linewidth=1,
        label=f"Initial k = {full_result['sol_init'].n_stations}",
    )
    ax_k.legend(loc="upper right", fontsize=9)
    fig_k.tight_layout()
    fig_k.savefig(
        str(_RESULTS_DIR / "q1c_alns_k_evolution_1072.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_k)

    fig_c, ax_c = plt.subplots(figsize=(12, 4))
    ax_c.plot(iters, cost_hist, linewidth=0.6, color="#2ECC71", alpha=0.9)
    ax_c.set_xlabel("Iteration")
    ax_c.set_ylabel("Best total cost (£)")
    ax_c.set_title(
        "Q1c ALNS+LAHC — Best total cost vs iteration (1072 robots)"
    )
    ax_c.grid(True, alpha=0.3)
    fig_c.tight_layout()
    fig_c.savefig(
        str(_RESULTS_DIR / "q1c_alns_best_cost_1072.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig_c)
    print(
        f"  Figures: {_RESULTS_DIR / 'q1c_alns_k_evolution_1072.png'}, "
        f"{_RESULTS_DIR / 'q1c_alns_best_cost_1072.png'}"
    )

    # ── Comparison: Q1a vs Q1b vs ALNS+LAHC (Q1a/Q1b from log txt) ─────
    print(f"\n{'=' * 90}")
    print("Q1c Comparison: Q1a (MINLP) vs Q1b vs ALNS+LAHC")
    print("  Q1a: results/q1a_output.txt (Haversine row in MINLP summary)")
    print("  Q1b: results/q1b_output.txt (TOTAL per construction block)")
    print("=" * 90)

    q1a_txt = _RESULTS_DIR / "q1a_output.txt"
    q1b_txt = _RESULTS_DIR / "q1b_output.txt"
    q1a_costs = parse_q1a_costs_from_output_txt(q1a_txt)
    q1b_costs = parse_q1b_costs_from_output_txt(q1b_txt)

    print(f"{'Instance':<30} {'Q1a':>12} {'Q1b':>12} "
          f"{'ALNS':>12} {'ALNS Improv':>12}")
    print("-" * 85)

    for label, r in small_results.items():
        alns_cost = r["cost"].total
        q1b_val = q1b_costs.get(label, r["init_cost"].total)
        q1a_str = "N/A"
        if label in q1a_costs:
            q1a_str = f"£{q1a_costs[label]:>10,.0f}"
        q1b_str = f"£{q1b_val:>10,.0f}"
        if q1b_val > 0:
            improv = (q1b_val - alns_cost) / q1b_val * 100
        else:
            improv = r["improvement"]
        print(f"{label:<30} {q1a_str:>12} {q1b_str:>12} "
              f"£{alns_cost:>10,.0f} {improv:>11.1f}%")

    label_full = "1072 robots (full)"
    row_label = f"{label_full}, k={FULL_K}"
    alns_cost = full_result["cost"].total
    q1b_val = q1b_costs.get(label_full, full_result["init_cost"].total)
    q1a_str = "N/A"
    if label_full in q1a_costs:
        q1a_str = f"£{q1a_costs[label_full]:>10,.0f}"
    q1b_str = f"£{q1b_val:>10,.0f}"
    if q1b_val > 0:
        improv = (q1b_val - alns_cost) / q1b_val * 100
    else:
        improv = full_result["improvement"]
    print(f"{row_label:<30} {q1a_str:>12} {q1b_str:>12} "
          f"£{alns_cost:>10,.0f} {improv:>11.1f}%")

    # Fig-C4: Cost comparison (full instance)
    cost_dict = {
        "Q1b": full_result["init_cost"],
        "ALNS": full_result["cost"],
    }
 
    fig, _ = plot_cost_comparison(
        cost_dict,
        title="Q1c Cost Comparison — 1072 robots (Fig-C4)",
        save_path=str(_RESULTS_DIR / "q1c_cost_comparison_1072.png"))
    plt.close(fig)
 
    # ── Full instance summary ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Q1c ALNS Full Instance Summary")
    print(f"{'=' * 60}")
    print(f"  Initial cost:   £{full_result['init_cost'].total:,.2f}")
    print(f"  ALNS cost:      £{full_result['cost'].total:,.2f}")
    print(f"  Improvement:    {full_result['improvement']:.2f}%")
    print(f"  Stations:       {full_result['sol'].n_stations}")
    print(f"  Total chargers: "
          f"{int(np.sum(full_result['sol'].n_chargers))}")
    print(f"  Transport:      "
          f"{int(np.sum(full_result['sol'].needs_transport))} robots")
    print(f"  Runtime:        {full_result['stats']['time']:.3f}s")
    print(f"  Feasible:       {full_result['feasible']}")
 
    # ── Save JSON ──────────────────────────────────────────────────────
    summary = {}
    for label, r in small_results.items():
        summary[label] = {
            "max_iter": SMALL_MAX_ITER,
            "time": r["stats"]["time"],
            "feasible": r["feasible"],
            "init_cost": r["init_cost"].as_dict(),
            "cost": r["cost"].as_dict(),
            "improvement": r["improvement"],
            "n_stations": r["sol"].n_stations,
            "n_chargers": r["sol"].n_chargers.tolist(),
            "transport_count": int(np.sum(r["sol"].needs_transport)),
            "operator_stats": r["stats"]["operator_stats"],
        }

    summary["1072 robots (full)"] = {
        "construct_k": FULL_K,
        "max_iter": FULL_MAX_ITER,
        "L": FULL_LAHC_L,
        "time": full_result["stats"]["time"],
        "feasible": full_result["feasible"],
        "init_cost": full_result["init_cost"].as_dict(),
        "cost": full_result["cost"].as_dict(),
        "improvement": full_result["improvement"],
        "n_stations": full_result["sol"].n_stations,
        "n_chargers": full_result["sol"].n_chargers.tolist(),
        "transport_count": int(np.sum(full_result["sol"].needs_transport)),
        "operator_stats": full_result["stats"]["operator_stats"],
    }

    with open(_RESULTS_DIR / "q1c_alns_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {_RESULTS_DIR / 'q1c_alns_results.json'}")

    print(f"\nOutput log saved to {_RESULTS_DIR / 'q1c_alns_output.txt'}")
    tee.close()


if __name__ == "__main__":
    main()