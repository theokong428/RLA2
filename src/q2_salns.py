"""
Q2 track B: Stochastic ALNS (SALNS).

ALNS edits first-stage site layout; objective is expected cost over scenarios
(per-scenario greedy recourse). Reuses Q1c operators D1–D3, R1–R3, L1–L3.
Includes CVaR / VSS-style reporting.

Usage:
    cd src && python q2_salns.py
"""

import numpy as np
import time
import json
import sys
import os
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, ".")

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

from data_loader import load_data, get_subset
from solution import (
    Solution, CostBreakdown, evaluate, check_feasibility,
    update_transport_flags, update_charger_counts, remove_empty_stations,
)
from utils import distance_matrix, haversine, charging_probability
from q1b_construction import construct_solution, kmedoids

# Q1c operators (shared with deterministic ALNS+LAHC)
from q1c_alns import (
    destroy_random_remove, destroy_zone_remove,
    repair_greedy_insert, repair_regret_insert, repair_cluster,
    local_station_relocate, local_charger_adjust, local_robot_swap,
    _compact_stations, _fix_overflow,
)
from q2_utils import evaluate_stochastic, compute_cvar


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
# Stochastic evaluation (second stage: per-scenario greedy assignment)
# ============================================================================

def stochastic_fast_evaluate(sol, data, scenario_subset=None):
    """Expected total cost: fixed first stage + mean second-stage cost per scenario.

    Returns (expected_total, second_stage_costs_per_evaluated_scenario).
    """
    # First stage
    first_stage = data.cb * sol.n_stations + data.cm * float(
        np.sum(sol.n_chargers))

    # Robot–station distances
    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )

    # Capacities
    capacity = sol.n_chargers * data.q  # (n_stations,)

    # Scenario indices
    if scenario_subset is not None:
        scenarios = scenario_subset
    else:
        scenarios = np.arange(data.n_scenarios)

    n_eval = len(scenarios)
    scenario_costs = np.zeros(n_eval)

    for idx, s in enumerate(scenarios):
        ranges_s = data.scenario_range[:, s]
        p_s = charging_probability(ranges_s, data.lam, data.r_min)

        # Greedy feasible assignment (independent per scenario)
        counts = np.zeros(sol.n_stations, dtype=int)
        total_cost_s = 0.0

        nearest = np.argmin(dmat, axis=1)
        order = np.argsort(dmat[np.arange(data.n_robots), nearest])

        for i in order:
            assigned = False
            for j in np.argsort(dmat[i]):
                if j >= sol.n_stations:
                    break
                if counts[j] < capacity[j]:
                    d = dmat[i, j]
                    transport = 1.0 if d > ranges_s[i] else 0.0
                    fly = 1.0 - transport
                    total_cost_s += p_s[i] * (
                        data.cc * d * fly + data.ch * transport)
                    counts[j] += 1
                    assigned = True
                    break

            if not assigned:
                # Overflow: assign to nearest (may exceed capacity)
                j = nearest[i]
                d = dmat[i, j]
                transport = 1.0 if d > ranges_s[i] else 0.0
                fly = 1.0 - transport
                total_cost_s += p_s[i] * (data.cc * d * fly + data.ch * transport)
                counts[j] += 1

        scenario_costs[idx] = total_cost_s

    expected_second = float(np.mean(scenario_costs))
    return first_stage + expected_second, scenario_costs


# ============================================================================
# D2 (stochastic): worst-remove by expected cost share
# ============================================================================

def destroy_worst_remove_stochastic(sol, data, rng, degree=None,
                                     scenario_subset=None):
    """Remove stations with largest expected cost contribution (sampled scenarios)."""
    if sol.n_stations < 3:
        return set()
    k = degree or max(1, rng.integers(1, max(2, sol.n_stations // 6)))

    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )

    # Expected station costs over scenario subset
    if scenario_subset is None:
        scenarios = rng.choice(data.n_scenarios, size=min(20, data.n_scenarios),
                               replace=False)
    else:
        scenarios = scenario_subset

    station_costs = np.zeros(sol.n_stations)
    for j in range(sol.n_stations):
        members = np.where(sol.assignments == j)[0]
        if len(members) == 0:
            continue

        station_costs[j] = data.cb + data.cm * sol.n_chargers[j]

        # Mean operating cost over scenarios
        for s in scenarios:
            ranges_s = data.scenario_range[members, s]
            p_s = charging_probability(ranges_s, data.lam, data.r_min)
            dists = dmat[members, j]
            transport = (dists > ranges_s).astype(float)
            fly = 1.0 - transport
            station_costs[j] += float(np.sum(
                p_s * (data.cc * dists * fly + data.ch * transport)
            )) / len(scenarios)

    worst_order = np.argsort(-station_costs)
    remove_ids = worst_order[:min(k, sol.n_stations - 1)]

    unassigned = set()
    for j in remove_ids:
        members = np.where(sol.assignments == j)[0]
        unassigned.update(members.tolist())
        sol.assignments[members] = -1
    return unassigned


# ============================================================================
# Operator lists
# ============================================================================

DESTROY_OPS = [
    ("D1: Random-Remove", destroy_random_remove),
    ("D2: Worst-Remove (stoch)", destroy_worst_remove_stochastic),
    ("D3: Zone-Remove", destroy_zone_remove),
]

REPAIR_OPS = [
    ("R1: Greedy-Insert", repair_greedy_insert),
    ("R2: Regret-Insert", repair_regret_insert),
    ("R3: Cluster-Repair", repair_cluster),
]

LOCAL_OPS = [
    ("L1: Station-Relocate", local_station_relocate),
    ("L2: Charger-Adjust", local_charger_adjust),
    ("L3: Robot-Swap", local_robot_swap),
]


# ============================================================================
# SALNS + LAHC loop
# ============================================================================

def salns_lahc(sol_init, data, max_iter=5000, L=300,
               segment_length=100, reaction_factor=0.1,
               sigma1=33, sigma2=9, sigma3=1,
               p_large=0.4, n_scenario_eval=None,
               rng=None, verbose=True):
    """Same structure as Q1c ALNS+LAHC; cost from ``stochastic_fast_evaluate``."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Optional scenario subsample
    scenario_subset = None
    if n_scenario_eval is not None and n_scenario_eval < data.n_scenarios:
        scenario_subset = rng.choice(data.n_scenarios, size=n_scenario_eval,
                                     replace=False)

    # Operator weights
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

    d_total_uses = np.zeros(n_d, dtype=int)
    r_total_uses = np.zeros(n_r, dtype=int)
    l_total_uses = np.zeros(n_l, dtype=int)
    d_total_improves = np.zeros(n_d, dtype=int)
    r_total_improves = np.zeros(n_r, dtype=int)
    l_total_improves = np.zeros(n_l, dtype=int)

    n_segments = max_iter // segment_length + 1
    d_weight_hist = np.zeros((n_segments, n_d))
    r_weight_hist = np.zeros((n_segments, n_r))
    l_weight_hist = np.zeros((n_segments, n_l))
    seg_idx = 0

    # Init
    current = sol_init.copy()
    current_cost, current_sc = stochastic_fast_evaluate(
        current, data, scenario_subset)
    best = current.copy()
    best_cost = current_cost
    best_scenario_costs = current_sc.copy()

    # LAHC history
    cost_history = np.full(L, current_cost)
    history = np.zeros(max_iter)

    t0 = time.time()

    for it in range(max_iter):
        if rng.random() < p_large:
            # === Destroy + Repair ===
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

            _compact_stations(candidate)
            update_charger_counts(candidate, data)

            # Stochastic evaluate
            cand_cost, cand_sc = stochastic_fast_evaluate(
                candidate, data, scenario_subset)

            lahc_ref = cost_history[it % L]
            old_cost = current_cost
            if cand_cost < lahc_ref or cand_cost < current_cost:
                current = candidate
                current_cost = cand_cost
                cost_history[it % L] = current_cost

                if cand_cost < best_cost - 1e-6:
                    best = current.copy()
                    best_cost = cand_cost
                    best_scenario_costs = cand_sc.copy()
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
                continue

            cand_cost, cand_sc = stochastic_fast_evaluate(
                candidate, data, scenario_subset)

            lahc_ref = cost_history[it % L]
            old_cost = current_cost
            if cand_cost < lahc_ref or cand_cost < current_cost:
                current = candidate
                current_cost = cand_cost
                cost_history[it % L] = current_cost

                if cand_cost < best_cost - 1e-6:
                    best = current.copy()
                    best_cost = cand_cost
                    best_scenario_costs = cand_sc.copy()
                    l_scores[l_idx] += sigma1
                    l_total_improves[l_idx] += 1
                elif cand_cost < old_cost - 1e-6:
                    l_scores[l_idx] += sigma2
                else:
                    l_scores[l_idx] += sigma3
            else:
                cost_history[it % L] = current_cost

        history[it] = best_cost

        # Weight update
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

            d_scores[:] = 0; r_scores[:] = 0; l_scores[:] = 0
            d_uses[:] = 0; r_uses[:] = 0; l_uses[:] = 0

        if verbose and (it + 1) % max(1, max_iter // 10) == 0:
            elapsed = time.time() - t0
            print(f"  SALNS iter {it+1:>6}/{max_iter}: "
                  f"best=£{best_cost:,.0f}  curr=£{current_cost:,.0f}  "
                  f"stations={current.n_stations}  time={elapsed:.1f}s")

    elapsed = time.time() - t0

    # Clean best solution
    remove_empty_stations(best)
    update_charger_counts(best, data)
    update_transport_flags(best, data)

    if scenario_subset is not None:
        print("  Re-evaluating best on all scenarios...")
        best_cost, best_scenario_costs = stochastic_fast_evaluate(
            best, data, scenario_subset=None)

    # Weight history dict
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
        "n_scenario_eval": n_scenario_eval or data.n_scenarios,
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
            "uses": int(uses), "improves": int(improves),
        }

    return best, best_cost, history, weight_history, stats, best_scenario_costs


# ============================================================================
# Main
# ============================================================================

def main():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_RESULTS_DIR / "q2_salns_output.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    print("Q2: Stochastic ALNS (SALNS)")
    print("=" * 70)

    data = load_data()
    print(f"Loaded: {data.n_robots} robots, {data.n_scenarios} scenarios\n")

    rng = np.random.default_rng(42)

    # ==================================================================
    # PART 1: Small instances
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: Small instances — SALNS")
    print("=" * 70)

    small_sizes = [20, 50, 100]
    rng_subset = np.random.default_rng(42)
    small_results = {}

    for n_robots in small_sizes:
        idx = rng_subset.choice(data.n_robots, n_robots, replace=False)
        sub = get_subset(data, idx)
        label = f"{n_robots} robots"

        print(f"\n--- {label} ---")

        sol_init = construct_solution(sub, rng=np.random.default_rng(42))
        init_cost, _ = stochastic_fast_evaluate(sol_init, sub)
        print(f"  Initial expected cost: £{init_cost:,.2f} "
              f"({sol_init.n_stations} stations)")
        best, best_cost, hist, wh, stats, sc = salns_lahc(
            sol_init, sub, max_iter=3000,
            rng=np.random.default_rng(456),
        )

        cvar, var = compute_cvar(sc, 0.95)
        print(f"  SALNS:")
        print(f"    Expected cost:  £{best_cost:,.2f}")
        print(f"    Stations:       {best.n_stations}")
        print(f"    Chargers:       {int(np.sum(best.n_chargers))}")
        print(f"    CVaR_0.95:      £{cvar:,.2f}")
        print(f"    VaR_0.95:       £{var:,.2f}")
        print(f"    Improvement:    "
              f"{(init_cost - best_cost) / init_cost * 100:.2f}%")
        print(f"    Time:           {stats['time']:.1f}s")

        small_results[label] = {
            "best_cost": best_cost,
            "init_cost": init_cost,
            "n_stations": best.n_stations,
            "n_chargers": int(np.sum(best.n_chargers)),
            "cvar": cvar,
            "var": var,
            "time": stats["time"],
            "improvement": (init_cost - best_cost) / init_cost * 100,
            "scenario_costs": sc,
            "history": hist,
            "weight_history": wh,
            "stats": stats,
            "sol": best,
        }

    # ==================================================================
    # PART 2: Full instance
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 2: Full instance SALNS (1072 robots x 100 scenarios)")
    print("=" * 70)

    sol_init_full = construct_solution(data, rng=np.random.default_rng(42))
    init_cost_full, init_sc = stochastic_fast_evaluate(sol_init_full, data)
    print(f"  Initial expected cost: £{init_cost_full:,.2f} "
          f"({sol_init_full.n_stations} stations)")

    print(f"\n  Running SALNS (30-scenario subsample, 5000 iters)...")
    best_full, best_cost_full, hist_full, wh_full, stats_full, sc_full = \
        salns_lahc(
            sol_init_full, data,
            max_iter=5000,
            n_scenario_eval=30,
            rng=np.random.default_rng(456),
        )

    cvar_full, var_full = compute_cvar(sc_full, 0.95)
    improvement_full = (init_cost_full - best_cost_full) / init_cost_full * 100

    print(f"\n  SALNS (full):")
    print(f"    Expected cost:  £{best_cost_full:,.2f}")
    print(f"    Stations:       {best_full.n_stations}")
    print(f"    Chargers:       {int(np.sum(best_full.n_chargers))}")
    print(f"    Transport:    {int(np.sum(best_full.needs_transport))} robots")
    print(f"    CVaR_0.95:      £{cvar_full:,.2f}")
    print(f"    VaR_0.95:       £{var_full:,.2f}")
    print(f"    Improvement:    {improvement_full:.2f}%")
    print(f"    Time:           {stats_full['time']:.1f}s")

    # ==================================================================
    # PART 3: VSS (vs deterministic Q1)
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 3: VSS — SALNS vs deterministic Q1")
    print("=" * 70)
    eev_result = evaluate_stochastic(data, sol_init_full)
    eev_cost = eev_result["expected_cost"]
    cvar_eev, var_eev = compute_cvar(eev_result["scenario_costs"], 0.95)

    rp_cost = best_cost_full

    vss = eev_cost - rp_cost

    print(f"  EEV (det. Q1, scenarios): £{eev_cost:,.2f}")
    print(f"  RP  (SALNS):              £{rp_cost:,.2f}")
    print(f"  VSS = EEV - RP:          £{vss:,.2f}")
    if eev_cost > 0:
        print(f"  VSS / EEV:               {vss / eev_cost * 100:.2f}%")
    print(f"\n  Det. CVaR_0.95:          £{cvar_eev:,.2f}")
    print(f"  SALNS CVaR_0.95:         £{cvar_full:,.2f}")
    print(f"  CVaR delta:              £{cvar_eev - cvar_full:,.2f}")

    # ==================================================================
    # PART 4: Operator stats
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 4: SALNS operator stats")
    print("=" * 70)

    for name, s in stats_full["operator_stats"].items():
        if s["uses"] > 0:
            rate = s["improves"] / s["uses"] * 100
            print(f"  {name:<35} uses={s['uses']:>5}  "
                  f"improves={s['improves']:>4}  rate={rate:.1f}%")

    # ==================================================================
    # PART 5: Figures
    # ==================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Convergence
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_full[:stats_full["iterations"]], linewidth=0.8,
            color="#3498DB")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Expected Total Cost (£)")
    ax.set_title("Q2 SALNS Convergence — 1072 robots × 100 scenarios")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        str(_RESULTS_DIR / "q2_salns_convergence.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Scenario cost histograms
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(eev_result["scenario_costs"], bins=25, alpha=0.7,
             color="#E67E22", edgecolor="white")
    ax1.axvline(np.mean(eev_result["scenario_costs"]), color="#E74C3C",
                linewidth=2, label=f"E=£{np.mean(eev_result['scenario_costs']):,.0f}")
    ax1.axvline(cvar_eev, color="#E74C3C", linewidth=2, linestyle="--",
                label=f"CVaR=£{cvar_eev:,.0f}")
    ax1.set_xlabel("Second-Stage Cost (£)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Q1 Deterministic Solution (EEV)")
    ax1.legend(fontsize=8)

    ax2.hist(sc_full, bins=25, alpha=0.7,
             color="#3498DB", edgecolor="white")
    ax2.axvline(np.mean(sc_full), color="#E74C3C",
                linewidth=2, label=f"E=£{np.mean(sc_full):,.0f}")
    ax2.axvline(cvar_full, color="#E74C3C", linewidth=2, linestyle="--",
                label=f"CVaR=£{cvar_full:,.0f}")
    ax2.set_xlabel("Second-Stage Cost (£)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Q2 SALNS Stochastic Solution (RP)")
    ax2.legend(fontsize=8)

    # Align x-axis
    x_min = min(ax1.get_xlim()[0], ax2.get_xlim()[0])
    x_max = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)

    fig.suptitle(f"Q2 Scenario Cost Distribution — VSS=£{vss:,.0f}", y=1.02)
    fig.tight_layout()
    fig.savefig(
        str(_RESULTS_DIR / "q2_salns_scenario_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Operator weights
    if wh_full:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, prefix, title in zip(
                axes, ["D", "R", "L"],
                ["Destroy", "Repair", "Local"]):
            for name, vals in wh_full.items():
                if name.startswith(prefix):
                    ax.plot(vals, label=name, linewidth=1.2)
            ax.set_xlabel("Segment")
            ax.set_ylabel("Weight")
            ax.set_title(f"{title} Operators")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Q2 SALNS Operator Weight Evolution — 1072 robots")
        fig.tight_layout()
        fig.savefig(
            str(_RESULTS_DIR / "q2_salns_operator_weights.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # VSS bar chart
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["EEV\n(Det. sol.)", "RP\n(SALNS)"]
    values = [eev_cost, rp_cost]
    colours = ["#E67E22", "#3498DB"]
    bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"£{val:,.0f}", ha="center", va="bottom", fontsize=10)

    ax.annotate("", xy=(0, eev_cost * 0.99), xytext=(1, rp_cost * 1.01),
                arrowprops=dict(arrowstyle="<->", color="#E74C3C", linewidth=2))
    ax.text(0.5, (eev_cost + rp_cost) / 2,
            f"VSS=£{vss:,.0f}\n({vss/eev_cost*100:.1f}%)",
            ha="center", color="#E74C3C", fontweight="bold", fontsize=11)

    ax.set_ylabel("Expected Total Cost (£)")
    ax.set_title("Q2 Value of Stochastic Solution (SALNS)")
    fig.tight_layout()
    fig.savefig(
        str(_RESULTS_DIR / "q2_salns_vss.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ==================================================================
    # Save JSON
    # ==================================================================
    summary = {
        "method": "Stochastic ALNS (SALNS)",
        "small_instances": {},
        "full_instance": {
            "expected_cost": best_cost_full,
            "init_cost": init_cost_full,
            "improvement": improvement_full,
            "n_stations": best_full.n_stations,
            "n_chargers": int(np.sum(best_full.n_chargers)),
            "transport_count": int(np.sum(best_full.needs_transport)),
            "cvar_095": cvar_full,
            "var_095": var_full,
            "time": stats_full["time"],
            "iterations": stats_full["iterations"],
            "n_scenario_eval": stats_full["n_scenario_eval"],
            "operator_stats": stats_full["operator_stats"],
        },
        "vss_analysis": {
            "eev": eev_cost,
            "rp": rp_cost,
            "vss": vss,
            "vss_pct": vss / eev_cost * 100 if eev_cost > 0 else 0,
            "eev_cvar_095": cvar_eev,
            "rp_cvar_095": cvar_full,
        },
    }

    for label, r in small_results.items():
        summary["small_instances"][label] = {
            "expected_cost": r["best_cost"],
            "init_cost": r["init_cost"],
            "improvement": r["improvement"],
            "n_stations": r["n_stations"],
            "n_chargers": r["n_chargers"],
            "cvar_095": r["cvar"],
            "time": r["time"],
        }

    with open(_RESULTS_DIR / "q2_salns_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {_RESULTS_DIR / 'q2_salns_results.json'}")
    print(f"Log:   {_RESULTS_DIR / 'q2_salns_output.txt'}")

    tee.close()


if __name__ == "__main__":
    main()
