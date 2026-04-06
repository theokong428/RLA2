"""
Q1b: Construction heuristic for Antarctic charging station placement.

Two-phase greedy approach:
  Phase 1 — k-medoids clustering (Haversine distance) to determine
            station locations from robot positions.
  Phase 2 — Greedy nearest-feasible assignment respecting the capacity
            constraint  n_chargers[j] * q  per station.

Tested on the same small subsets as Q1a (for gap comparison) and the
full 1072-robot instance.

Usage:
    python q1b_construction.py   # from src/
    python src/q1b_construction.py   # from project root (outputs still go to project results/)
"""

import numpy as np
import time
import json
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

from data_loader import load_data, get_subset
from solution import (
    Solution, evaluate, check_feasibility,
    update_transport_flags, update_charger_counts, remove_empty_stations,
)
from utils import distance_matrix


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
# K-medoids (PAM) clustering
# ============================================================================

def kmedoids(dist_mat, k, max_iter=100, rng=None):
    """K-medoids clustering with farthest-first initialisation.

    Args:
        dist_mat: (n, n) symmetric pairwise distance matrix.
        k:        Number of clusters.
        max_iter: Maximum swap iterations.
        rng:      numpy random generator.

    Returns:
        medoids: (k,) indices of medoid points.
        labels:  (n,) cluster assignment for each point.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = dist_mat.shape[0]
    if k >= n:
        return np.arange(n), np.arange(n)

    # ── Farthest-first initialisation ──────────────────────────────────
    medoids = np.empty(k, dtype=int)
    medoids[0] = rng.integers(n)

    for i in range(1, k):
        d_nearest = dist_mat[:, medoids[:i]].min(axis=1)
        medoids[i] = np.argmax(d_nearest)

    # ── PAM swap iterations ────────────────────────────────────────────
    for _ in range(max_iter):
        # Assign to nearest medoid
        d_to_med = dist_mat[:, medoids]          # (n, k)
        labels = np.argmin(d_to_med, axis=1)

        # Update each medoid to the cluster member with lowest total
        # intra-cluster distance
        changed = False
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            total_d = dist_mat[np.ix_(members, members)].sum(axis=1)
            best = members[np.argmin(total_d)]
            if best != medoids[c]:
                medoids[c] = best
                changed = True

        if not changed:
            break

    # Final assignment
    labels = np.argmin(dist_mat[:, medoids], axis=1)
    return medoids, labels


# ============================================================================
# Greedy construction
# ============================================================================

def construct_solution(data, k=None, rng=None):
    """Build a feasible solution via k-medoids + greedy assignment.

    Steps
    -----
    1. Compute full robot–robot Haversine distance matrix.
    2. Run k-medoids to obtain k station locations (at medoid robots).
    3. Greedy nearest-feasible assignment: each robot is assigned to the
       nearest station that still has capacity (≤ m*q robots).
    4. If any station is over-capacity, sub-cluster the overflow and add
       new stations.
    5. Remove empty stations, set charger counts and transport flags.

    Args:
        data: ProblemData.
        k:    Initial number of clusters.  Default = ceil(n / (m*q)).
        rng:  Random generator.

    Returns:
        Solution object.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = data.n_robots
    capacity = data.m * data.q          # 16 robots per station

    if k is None:
        k = max(int(np.ceil(n / capacity)), 1)

    # 1. Robot–robot distance matrix (Haversine, miles)
    robot_dm = distance_matrix(
        data.robot_lat, data.robot_lon,
        data.robot_lat, data.robot_lon,
    )

    # 2. K-medoids
    medoids, _ = kmedoids(robot_dm, k, rng=rng)

    station_lon = data.robot_lon[medoids].copy()
    station_lat = data.robot_lat[medoids].copy()

    # 3. Greedy nearest-feasible assignment
    rs_dm = distance_matrix(
        data.robot_lat, data.robot_lon,
        station_lat, station_lon,
    )

    assignments = np.full(n, -1, dtype=int)
    station_counts = np.zeros(len(station_lon), dtype=int)  # robots per station

    # Process robots sorted by distance to their closest station
    min_dists = rs_dm.min(axis=1)
    order = np.argsort(min_dists)

    for i in order:
        for j in np.argsort(rs_dm[i]):
            if station_counts[j] < capacity:
                assignments[i] = j
                station_counts[j] += 1
                break

    # 4. Handle any remaining unassigned robots (capacity exhausted)
    unassigned = np.where(assignments == -1)[0]
    if len(unassigned) > 0:
        # Add new stations at unassigned robot locations
        for i in unassigned:
            new_j = len(station_lon)
            station_lon = np.append(station_lon, data.robot_lon[i])
            station_lat = np.append(station_lat, data.robot_lat[i])
            station_counts = np.append(station_counts, 1)
            assignments[i] = new_j

    # 5. Build Solution, clean up
    sol = Solution(
        station_lon=station_lon,
        station_lat=station_lat,
        n_chargers=np.zeros(len(station_lon), dtype=int),
        assignments=assignments,
        needs_transport=np.zeros(n, dtype=bool),
    )
    remove_empty_stations(sol)
    update_charger_counts(sol, data)
    update_transport_flags(sol, data)

    return sol


# ============================================================================
# Solve wrapper
# ============================================================================

def solve_instance(data, k=None, label="", rng=None):
    """Construct a solution and report results."""
    print(f"\n{'=' * 60}")
    print(f"Q1b Construction: {label}")
    cap = data.m * data.q
    k_used = k if k else max(int(np.ceil(data.n_robots / cap)), 1)
    print(f"  n={data.n_robots}, k={k_used}, capacity/station={cap}")
    print(f"{'=' * 60}")

    t0 = time.time()
    sol = construct_solution(data, k=k, rng=rng)
    t_solve = time.time() - t0

    cost = evaluate(sol, data)
    feasible, violations = check_feasibility(sol, data)

    result = {
        "sol": sol, "cost": cost, "time": t_solve,
        "feasible": feasible,
    }

    print(f"  Time:      {t_solve:.3f}s")
    print(f"  Stations:  {sol.n_stations}")
    print(f"  Chargers:  {sol.n_chargers.tolist()}")
    print(f"  Transport: {int(np.sum(sol.needs_transport))} robots")
    print(f"\n  Cost breakdown:")
    print(f"    Build:       £{cost.build:,.2f}")
    print(f"    Maintenance: £{cost.maintenance:,.2f}")
    print(f"    Charging:    £{cost.charging:,.2f}")
    print(f"    Transport:   £{cost.transport:,.2f}")
    print(f"    TOTAL:       £{cost.total:,.2f}")
    print(f"  Feasible:  {feasible}")
    if not feasible:
        for v in violations:
            print(f"    - {v}")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_RESULTS_DIR / "q1b_output.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    print("Q1b: Construction Heuristic for Charging Station Placement")
    print("=" * 60)

    data = load_data()
    print(f"Full data loaded: {data.n_robots} robots\n")

    rng = np.random.default_rng(42)

    # ── Small instances (same subsets as Q1a for comparison) ────────────
    small_sizes = [20, 50, 100]
    rng_subset = np.random.default_rng(42)      # same seed as Q1a
    small_results = {}

    for n_robots in small_sizes:
        idx = rng_subset.choice(data.n_robots, n_robots, replace=False)
        sub = get_subset(data, idx)
        label = f"{n_robots} robots"

        r = solve_instance(sub, label=label, rng=rng)
        small_results[label] = r

        # Visualisation
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization import plot_solution_map

        fig, _ = plot_solution_map(
            r["sol"], sub,
            title=f"Q1b Construction — {label}",
            save_path=str(_RESULTS_DIR / f"q1b_map_{n_robots}.png"))
        plt.close(fig)

    # ── Full instance (1072 robots) ─────────────────────────────────────
    full_result = solve_instance(data, label="1072 robots (full)", rng=rng)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization import plot_solution_map, plot_cost_breakdown, plot_k_scan

    fig, _ = plot_solution_map(
        full_result["sol"], data,
        title="Q1b Construction — 1072 robots (Fig-B1)",
        save_path=str(_RESULTS_DIR / "q1b_map_1072.png"))
    plt.close(fig)

    fig, _ = plot_cost_breakdown(
        full_result["cost"],
        title="Q1b Cost Breakdown — 1072 robots (Fig-B2)",
        save_path=str(_RESULTS_DIR / "q1b_cost_1072.png"))
    plt.close(fig)

    # ── Comparison with Q1a MINLP ───────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("Q1b vs Q1a Comparison (small instances)")
    print("=" * 75)

    q1a_path = str(_RESULTS_DIR / "q1a_results.json")
    q1a_data = {}
    if os.path.exists(q1a_path):
        with open(q1a_path) as f:
            q1a_data = json.load(f)

    print(f"{'Instance':<15} {'Q1b Cost':>12} {'Q1a Cost':>12} "
          f"{'Gap':>8} {'Q1b Time':>10} {'Q1a Time':>10}")
    print("-" * 75)

    for label, r in small_results.items():
        q1b_cost = r["cost"].total
        q1b_time = r["time"]

        if label in q1a_data and "haversine_cost" in q1a_data[label]:
            q1a_cost = q1a_data[label]["haversine_cost"]["total"]
            q1a_time = q1a_data[label]["time"]
            gap = (q1b_cost - q1a_cost) / q1a_cost * 100
            print(f"{label:<15} £{q1b_cost:>10,.0f} £{q1a_cost:>10,.0f} "
                  f"{gap:>7.1f}% {q1b_time:>9.3f}s {q1a_time:>9.1f}s")
        else:
            print(f"{label:<15} £{q1b_cost:>10,.0f} {'N/A':>12} "
                  f"{'N/A':>8} {q1b_time:>9.3f}s {'N/A':>10}")

    # ── Full instance summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Q1b Full Instance Summary")
    print(f"{'=' * 60}")
    print(f"  Stations:       {full_result['sol'].n_stations}")
    print(f"  Total chargers: {int(np.sum(full_result['sol'].n_chargers))}")
    print(f"  Transport:      "
          f"{int(np.sum(full_result['sol'].needs_transport))} robots")
    print(f"  Total cost:     £{full_result['cost'].total:,.2f}")
    print(f"  Runtime:        {full_result['time']:.3f}s")
    print(f"  Feasible:       {full_result['feasible']}")

    # ── Save JSON ───────────────────────────────────────────────────────
    summary = {}
    for label, r in small_results.items():
        summary[label] = {
            "time": r["time"],
            "feasible": r["feasible"],
            "cost": r["cost"].as_dict(),
            "n_stations": r["sol"].n_stations,
            "n_chargers": r["sol"].n_chargers.tolist(),
            "transport_count": int(np.sum(r["sol"].needs_transport)),
            "station_lon": r["sol"].station_lon.tolist(),
            "station_lat": r["sol"].station_lat.tolist(),
        }

    summary["1072 robots (full)"] = {
        "time": full_result["time"],
        "feasible": full_result["feasible"],
        "cost": full_result["cost"].as_dict(),
        "n_stations": full_result["sol"].n_stations,
        "n_chargers": full_result["sol"].n_chargers.tolist(),
        "transport_count": int(np.sum(full_result["sol"].needs_transport)),
        "station_lon": full_result["sol"].station_lon.tolist(),
        "station_lat": full_result["sol"].station_lat.tolist(),
    }

    with open(_RESULTS_DIR / "q1b_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {_RESULTS_DIR / 'q1b_results.json'}")

    # ==================================================================
    # K-scan: sweep k with multiple seeds, parallel execution
    # ==================================================================
    print(f"\n{'=' * 75}")
    print("Q1b K-Scan: Station Count Sensitivity Analysis")
    print("=" * 75)

    K_VALUES = np.linspace(80, 170, 10, dtype=int)
    N_SEEDS = 10
    BASE_SEED = 100

    # Build argument list: (k, seed) pairs
    scan_args = [(int(k), BASE_SEED + s) for k in K_VALUES for s in range(N_SEEDS)]

    print(f"  k values:  {K_VALUES.tolist()}")
    print(f"  Seeds/k:   {N_SEEDS}")
    print(f"  Total runs: {len(scan_args)}")

    t_scan_start = time.time()
    with ProcessPoolExecutor() as pool:
        scan_results_raw = list(pool.map(_k_scan_worker, scan_args))
    t_scan = time.time() - t_scan_start
    print(f"  Parallel scan completed in {t_scan:.1f}s")

    # Aggregate: for each k, take the best (minimum total cost) across seeds
    k_best = {}       # k -> best result dict
    k_all_costs = {}  # k -> list of total costs (for stats)
    for (k_val, seed), res in zip(scan_args, scan_results_raw):
        if k_val not in k_best or res["total"] < k_best[k_val]["total"]:
            k_best[k_val] = res
        k_all_costs.setdefault(k_val, []).append(res["total"])

    # Build cost matrix for plotting
    k_sorted = sorted(k_best.keys())
    cost_matrix = {
        "total": [k_best[k]["total"] for k in k_sorted],
        "build": [k_best[k]["build"] for k in k_sorted],
        "maintenance": [k_best[k]["maintenance"] for k in k_sorted],
        "charging": [k_best[k]["charging"] for k in k_sorted],
        "transport": [k_best[k]["transport"] for k in k_sorted],
    }

    # Print table
    print(f"\n  {'k':>5} {'Best Cost':>12} {'Mean Cost':>12} {'Std':>10} "
          f"{'Stations':>8} {'Transport':>10}")
    print("  " + "-" * 65)
    for k in k_sorted:
        costs_arr = k_all_costs[k]
        print(f"  {k:>5} £{k_best[k]['total']:>10,.0f} "
              f"£{np.mean(costs_arr):>10,.0f} £{np.std(costs_arr):>8,.0f} "
              f"{k_best[k]['n_stations']:>8} "
              f"{k_best[k]['transport_count']:>10}")

    best_k = k_sorted[np.argmin(cost_matrix["total"])]
    print(f"\n  >>> Optimal k = {best_k}, "
          f"cost = £{k_best[best_k]['total']:,.0f}")

    # Plot k-scan curve
    fig, _ = plot_k_scan(
        np.array(k_sorted), cost_matrix, method_name="Q1b",
        title="Q1b Station Count Sensitivity (Fig-B3)",
        save_path=str(_RESULTS_DIR / "q1b_k_scan.png"))
    plt.close(fig)

    # Save k-scan results to JSON
    k_scan_summary = {
        "k_values": k_sorted,
        "n_seeds": N_SEEDS,
        "best_k": best_k,
        "scan_time": t_scan,
        "per_k": {
            str(k): {
                "best_cost": k_best[k],
                "mean_total": float(np.mean(k_all_costs[k])),
                "std_total": float(np.std(k_all_costs[k])),
            }
            for k in k_sorted
        },
    }
    with open(_RESULTS_DIR / "q1b_k_scan.json", "w", encoding="utf-8") as f:
        json.dump(k_scan_summary, f, indent=2)
    print(f"  K-scan results saved to {_RESULTS_DIR / 'q1b_k_scan.json'}")

    # Save the best-k full solution for Q1c to use
    best_k_sol = construct_solution(
        data, k=best_k,
        rng=np.random.default_rng(
            BASE_SEED + int(
                np.argmin(k_all_costs[best_k])
            )
        ),
    )
    best_k_cost = evaluate(best_k_sol, data)
    fig, _ = plot_solution_map(
        best_k_sol, data,
        title=f"Q1b Best-K Construction — k={best_k} (Fig-B4)",
        save_path=str(_RESULTS_DIR / "q1b_map_bestk.png"))
    plt.close(fig)

    print(f"\n  Best-K solution: k={best_k}, {best_k_sol.n_stations} stations, "
          f"cost=£{best_k_cost.total:,.0f}")

    # Update summary JSON with best-k and k-scan info
    summary["best_k"] = best_k
    summary[f"1072 robots (k={best_k})"] = {
        "time": k_best[best_k].get("time", 0),
        "feasible": True,
        "cost": best_k_cost.as_dict(),
        "n_stations": best_k_sol.n_stations,
        "n_chargers": best_k_sol.n_chargers.tolist(),
        "transport_count": int(np.sum(best_k_sol.needs_transport)),
    }
    with open(_RESULTS_DIR / "q1b_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutput log saved to {_RESULTS_DIR / 'q1b_output.txt'}")
    tee.close()


def _k_scan_worker(args):
    """Worker for parallel k-scan. Must be top-level for pickling."""
    k, seed = args
    # Re-import inside worker (separate process)
    from data_loader import load_data
    from solution import evaluate
    data = load_data()
    rng = np.random.default_rng(seed)
    sol = construct_solution(data, k=k, rng=rng)
    cost = evaluate(sol, data)
    return {
        "total": cost.total,
        "build": cost.build,
        "maintenance": cost.maintenance,
        "charging": cost.charging,
        "transport": cost.transport,
        "n_stations": sol.n_stations,
        "transport_count": int(np.sum(sol.needs_transport)),
        "n_chargers": sol.n_chargers.tolist(),
        "seed": seed,
        "time": 0,
    }


if __name__ == "__main__":
    main()
