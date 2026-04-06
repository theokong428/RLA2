"""
Q2 track A: SAA + Benders (full-instance oriented workflow).

k-medoids candidates, L-shaped Benders, EEV of Q1 deterministic solution,
VSS/EVPI, CVaR sensitivity. Differs from q2_stochastic.py in focus and knobs.

Usage:
    cd src && python q2_benders_full.py
"""

import numpy as np
import time
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, ".")

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

from data_loader import load_data, get_subset
from solution import (
    Solution, CostBreakdown, evaluate, check_feasibility,
    update_transport_flags, update_charger_counts, remove_empty_stations,
)
from utils import distance_matrix, charging_probability
from q1b_construction import construct_solution, kmedoids
from q2_benders import (
    BendersDecomposition,
    precompute_scenario_costs,
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
# Candidate sites (k-medoids on robot locations)
# ============================================================================

def generate_candidates(data, n_cand, rng=None):
    """Pick candidate site coordinates via k-medoids on robots."""
    if rng is None:
        rng = np.random.default_rng(42)

    if n_cand >= data.n_robots:
        return data.robot_lon.copy(), data.robot_lat.copy()

    dm = distance_matrix(
        data.robot_lat, data.robot_lon,
        data.robot_lat, data.robot_lon,
    )
    medoids, _ = kmedoids(dm, n_cand, rng=rng)
    return data.robot_lon[medoids].copy(), data.robot_lat[medoids].copy()


# ============================================================================
# Wait-and-see (WS)
# ============================================================================

def compute_ws(data, n_robots_ws=50, rng=None):
    """WS: per-scenario solve, mean cost (subset of n_robots_ws robots)."""
    if rng is None:
        rng = np.random.default_rng(42)

    idx = rng.choice(data.n_robots, n_robots_ws, replace=False)
    sub = get_subset(data, idx)

    n_s = sub.n_scenarios
    ws_costs = np.zeros(n_s)

    for s in range(n_s):
        from dataclasses import replace
        data_s = replace(sub, robot_range=sub.scenario_range[:, s].copy())
        sol_s = construct_solution(data_s, rng=np.random.default_rng(s))

        ranges_s = sub.scenario_range[:, s]
        p_s = charging_probability(ranges_s, sub.lam, sub.r_min)

        dmat = distance_matrix(
            sub.robot_lat, sub.robot_lon,
            sol_s.station_lat, sol_s.station_lon,
        )
        rd = dmat[np.arange(sub.n_robots), sol_s.assignments]
        transport = (rd > ranges_s).astype(float)

        first_stage = sub.cb * sol_s.n_stations + sub.cm * float(
            np.sum(sol_s.n_chargers))
        fly = 1.0 - transport
        second_stage = float(np.sum(
            p_s * (sub.cc * rd * fly + sub.ch * transport)))

        ws_costs[s] = first_stage + second_stage

    return sub, float(np.mean(ws_costs)), ws_costs


# ============================================================================
# Main
# ============================================================================

def main():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_RESULTS_DIR / "q2_benders_full_output.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    print("Q2: SAA + Benders decomposition")
    print("=" * 70)

    data = load_data()
    print(f"Loaded: {data.n_robots} robots, {data.n_scenarios} scenarios\n")

    rng = np.random.default_rng(42)
    all_results = {}

    # ==================================================================
    # PART 1: Small instances, risk-neutral Benders
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 1: Small instances — Benders (risk-neutral)")
    print("=" * 70)

    small_sizes = [20, 50, 100]
    rng_subset = np.random.default_rng(42)
    small_results = {}

    for n_robots in small_sizes:
        idx = rng_subset.choice(data.n_robots, n_robots, replace=False)
        sub = get_subset(data, idx)
        label = f"{n_robots} robots"

        cand_lon = sub.robot_lon.copy()
        cand_lat = sub.robot_lat.copy()

        print(f"\n--- {label}: {len(cand_lon)} candidate sites ---")

        bd = BendersDecomposition(
            sub, cand_lon, cand_lat,
            alpha_cvar=0.0, beta_cvar=0.95,
            time_limit=600,
        )
        result = bd.solve(max_iter=80, verbose=True)

        if result["y"] is not None:
            open_st = sum(1 for v in result["y"] if v > 0.5)
            total_ch = sum(result["n"][j] for j in range(len(result["n"]))
                          if result["y"][j] > 0.5)
            cvar, var = compute_cvar(result["scenario_costs"], 0.95)

            print(f"  Result: obj=£{result['objective']:,.2f}, "
                  f"stations={open_st}, chargers={total_ch}, "
                  f"gap={result['gap']:.4f}, time={result['time']:.1f}s")
            print(f"  CVaR_0.95=£{cvar:,.2f}, VaR_0.95=£{var:,.2f}")

            result["open_stations"] = open_st
            result["total_chargers"] = total_ch
            result["cvar"] = cvar
            result["var"] = var

        small_results[label] = result

    # # ==================================================================
    # # PART 2: Full instance Benders (k-medoids candidates) — optional
    # # ==================================================================
    # print("\n" + "=" * 70)
    # print("PART 2: Full instance Benders (k-medoids candidates)")
    # print("=" * 70)

    # # Vary candidate count
    # candidate_counts = [80, 120, 150]

    # for n_cand in candidate_counts:
    #     print(f"\n--- Candidate count: {n_cand} ---")
    #     cand_lon, cand_lat = generate_candidates(data, n_cand, rng=rng)
    #     print(f"  Generated {len(cand_lon)} candidates")

    #     bd = BendersDecomposition(
    #         data, cand_lon, cand_lat,
    #         alpha_cvar=0.0, beta_cvar=0.95,
    #         time_limit=1800,
    #     )
    #     result = bd.solve(max_iter=60, verbose=True)

    #     if result["y"] is not None:
    #         open_st = sum(1 for v in result["y"] if v > 0.5)
    #         total_ch = sum(result["n"][j] for j in range(len(result["n"]))
    #                       if result["y"][j] > 0.5)
    #         cvar, var = compute_cvar(result["scenario_costs"], 0.95)

    #         print(f"  Result: obj=£{result['objective']:,.2f}, "
    #               f"stations={open_st}, chargers={total_ch}")
    #         print(f"  gap={result['gap']:.4f}, time={result['time']:.1f}s")
    #         print(f"  CVaR_0.95=£{cvar:,.2f}")

    #         result["open_stations"] = open_st
    #         result["total_chargers"] = total_ch
    #         result["cvar"] = cvar
    #         result["var"] = var

    #     all_results[f"full_{n_cand}_cand"] = result

    # ==================================================================
    # PART 3: EEV — Q1 deterministic solution under scenarios
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 3: EEV — deterministic Q1 solution under scenarios")
    print("=" * 70)

    sol_det = construct_solution(data, rng=np.random.default_rng(42))
    det_cost = evaluate(sol_det, data)
    print(f"  Deterministic Q1 cost: £{det_cost.total:,.2f} "
          f"({sol_det.n_stations} stations)")

    print(f"  Evaluating over {data.n_scenarios} scenarios...")
    t0 = time.time()
    eev_result = evaluate_stochastic(data, sol_det)
    t_eev = time.time() - t0

    cvar_eev, var_eev = compute_cvar(eev_result["scenario_costs"], 0.95)

    print(f"  EEV:")
    print(f"    First stage:        £{eev_result['first_stage_cost']:,.2f}")
    print(f"    E[second stage]:    £{eev_result['expected_second_stage']:,.2f}")
    print(f"    Expected total:     £{eev_result['expected_cost']:,.2f}")
    print(f"    VaR_0.95:           £{var_eev:,.2f}")
    print(f"    CVaR_0.95:          £{cvar_eev:,.2f}")
    print(f"    Time:               {t_eev:.1f}s")

    # ==================================================================
    # PART 4: CVaR sensitivity
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 4: CVaR sensitivity (50-robot subset)")
    print("=" * 70)

    rng_cvar = np.random.default_rng(42)
    idx_50 = rng_cvar.choice(data.n_robots, 50, replace=False)
    sub_50 = get_subset(data, idx_50)
    cand_lon_50 = sub_50.robot_lon.copy()
    cand_lat_50 = sub_50.robot_lat.copy()

    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    cvar_results = {}

    for alpha in alpha_values:
        print(f"\n--- α={alpha} ---")
        bd = BendersDecomposition(
            sub_50, cand_lon_50, cand_lat_50,
            alpha_cvar=alpha, beta_cvar=0.95,
            time_limit=300,
        )
        r = bd.solve(max_iter=60, verbose=True)

        if r["y"] is not None:
            open_st = sum(1 for v in r["y"] if v > 0.5)
            total_ch = sum(r["n"][j] for j in range(len(r["n"]))
                          if r["y"][j] > 0.5)
            cvar_val, var_val = compute_cvar(r["scenario_costs"], 0.95)
            r["open_stations"] = open_st
            r["total_chargers"] = total_ch
            r["cvar"] = cvar_val
            r["var"] = var_val

        cvar_results[alpha] = r

    print(f"\n  CVaR sensitivity summary (50 robots, beta=0.95):")
    print(f"  {'alpha':>5}  {'obj':>12}  {'E[cost]':>12}  "
          f"{'CVaR':>12}  {'st':>6}  {'ch':>6}")
    print(f"  {'-' * 65}")
    for alpha, r in cvar_results.items():
        if r["y"] is None:
            print(f"  {alpha:>5.1f}  {'N/A':>12}")
            continue
        e_cost = r["first_stage_cost"] + float(
            np.mean(r["scenario_costs"]))
        print(f"  {alpha:>5.1f}  £{r['objective']:>10,.0f}  "
              f"£{e_cost:>10,.0f}  £{r.get('cvar', 0):>10,.0f}  "
              f"{r.get('open_stations', 0):>6}  "
              f"{r.get('total_chargers', 0):>6}")

    # ==================================================================
    # PART 5: VSS / EVPI
    # ==================================================================
    print("\n" + "=" * 70)
    print("PART 5: VSS / EVPI")
    print("=" * 70)

    # RP (50-robot)
    rp_50 = cvar_results.get(0.0, small_results.get("50 robots"))

    # EEV (50-robot)
    sol_det_50 = construct_solution(sub_50, rng=np.random.default_rng(42))
    eev_50 = evaluate_stochastic(sub_50, sol_det_50)
    eev_cost_50 = eev_50["expected_cost"]

    # WS (50-robot)
    print(f"  Computing wait-and-see (one problem per scenario)...")
    t0 = time.time()
    sub_ws, ws_cost, ws_scenario_costs = compute_ws(data, n_robots_ws=50, rng=rng)
    t_ws = time.time() - t0
    print(f"  WS time: {t_ws:.1f}s")

    print(f"\n  Value-of-information (50 robots):")
    print(f"  {'Metric':<30} {'Value':>12}")
    print(f"  {'-' * 45}")
    print(f"  {'WS (Wait-and-See)':<30} £{ws_cost:>10,.2f}")

    rp_cost = None
    if rp_50 and rp_50["y"] is not None:
        rp_cost = rp_50["objective"]
        print(f"  {'RP (Recourse Problem)':<30} £{rp_cost:>10,.2f}")
    print(f"  {'EEV (det. sol., scenarios)':<30} £{eev_cost_50:>10,.2f}")

    vss, evpi = None, None
    if rp_cost:
        vss = eev_cost_50 - rp_cost
        evpi = rp_cost - ws_cost
        print(f"\n  {'VSS = EEV - RP':<30} £{vss:>10,.2f}")
        print(f"  {'EVPI = RP - WS':<30} £{evpi:>10,.2f}")
        if eev_cost_50 > 0:
            print(f"  {'VSS / EEV':<30} {vss / eev_cost_50 * 100:>10.2f}%")
        if rp_cost > 0:
            print(f"  {'EVPI / RP':<30} {evpi / rp_cost * 100:>10.2f}%")

    # # Full-instance VSS (optional)
    # print(f"\n  Value-of-information (full 1072 robots):")
    # best_full_key = None
    # best_full_obj = np.inf
    # for key, r in all_results.items():
    #     if r["y"] is not None and r["objective"] < best_full_obj:
    #         best_full_obj = r["objective"]
    #         best_full_key = key

    # if best_full_key:
    #     rp_full = all_results[best_full_key]["objective"]
    #     eev_full = eev_result["expected_cost"]
    #     vss_full = eev_full - rp_full
    #     print(f"  {'RP (Benders, ' + best_full_key + ')':<30} £{rp_full:>10,.2f}")
    #     print(f"  {'EEV (Q1b deterministic)':<30} £{eev_full:>10,.2f}")
    #     print(f"  {'VSS = EEV - RP':<30} £{vss_full:>10,.2f}")
    # else:
    #     print(f"  {'EEV (Q1b deterministic)':<30} "
    #           f"£{eev_result['expected_cost']:>10,.2f}")
    #     print("  (full Benders RP not available)")

    # ==================================================================
    # PART 6: Figures
    # ==================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Scenario cost histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(eev_result["scenario_costs"], bins=30, alpha=0.7,
            color="#3498DB", edgecolor="white", label="Scenario costs")
    ax.axvline(np.mean(eev_result["scenario_costs"]), color="#E74C3C",
               linewidth=2, label=f"E[cost]=£{np.mean(eev_result['scenario_costs']):,.0f}")
    ax.axvline(var_eev, color="#F39C12", linewidth=2, linestyle="--",
               label=f"VaR_0.95=£{var_eev:,.0f}")
    ax.axvline(cvar_eev, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"CVaR_0.95=£{cvar_eev:,.0f}")
    ax.set_xlabel("Second-Stage Cost (£)")
    ax.set_ylabel("Frequency")
    ax.set_title("Q2 Benders: Scenario Cost Distribution — 1072 robots")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        str(_RESULTS_DIR / "q2_benders_scenario_dist.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # CVaR sensitivity plots
    alphas, objs, cvars, stations_list = [], [], [], []
    for alpha, r in cvar_results.items():
        if r["y"] is not None:
            alphas.append(alpha)
            objs.append(r["objective"])
            cvars.append(r.get("cvar", 0))
            stations_list.append(r.get("open_stations", 0))

    if alphas:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(alphas, objs, "o-", color="#3498DB", linewidth=2)
        ax1.set_xlabel("Risk weight α")
        ax1.set_ylabel("Objective (£)")
        ax1.set_title("CVaR Sensitivity: Objective vs α")
        ax1.grid(True, alpha=0.3)

        ax2.bar(alphas, stations_list, width=0.08, color="#2ECC71",
                edgecolor="white")
        ax2.set_xlabel("Risk weight α")
        ax2.set_ylabel("Open Stations")
        ax2.set_title("CVaR Sensitivity: Stations vs α")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(
            str(_RESULTS_DIR / "q2_benders_cvar_sensitivity.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # VSS / EVPI bar chart
    if vss is not None and evpi is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ["WS", "RP", "EEV"]
        values = [ws_cost, rp_cost, eev_cost_50]
        colours = ["#2ECC71", "#3498DB", "#E67E22"]
        bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"£{val:,.0f}", ha="center", va="bottom", fontsize=9)

        mid_rp_eev = (rp_cost + eev_cost_50) / 2
        ax.annotate("", xy=(1, rp_cost), xytext=(2, eev_cost_50),
                    arrowprops=dict(arrowstyle="<->", color="#E74C3C", linewidth=2))
        ax.text(1.5, mid_rp_eev, f"VSS=£{vss:,.0f}",
                ha="center", color="#E74C3C", fontweight="bold")

        mid_ws_rp = (ws_cost + rp_cost) / 2
        ax.annotate("", xy=(0, ws_cost), xytext=(1, rp_cost),
                    arrowprops=dict(arrowstyle="<->", color="#8E44AD", linewidth=2))
        ax.text(0.5, mid_ws_rp, f"EVPI=£{evpi:,.0f}",
                ha="center", color="#8E44AD", fontweight="bold")

        ax.set_ylabel("Expected Total Cost (£)")
        ax.set_title("Q2 Benders: VSS / EVPI — 50 robots")
        fig.tight_layout()
        fig.savefig(
            str(_RESULTS_DIR / "q2_benders_vss_evpi.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Benders convergence (optional)
    # for key, r in all_results.items():
    #     if r.get("history"):
    #         hist = r["history"]
    #         iters = [h["iter"] for h in hist]
    #         lbs = [h["LB"] for h in hist]
    #         ubs = [h["UB"] for h in hist]

    #         fig, ax = plt.subplots(figsize=(10, 5))
    #         ax.plot(iters, lbs, "b-", linewidth=1.5, label="Lower Bound")
    #         ub_finite = [(i, u) for i, u in zip(iters, ubs) if u < 1e15]
    #         if ub_finite:
    #             ax.plot([x[0] for x in ub_finite],
    #                     [x[1] for x in ub_finite],
    #                     "r-", linewidth=1.5, label="Upper Bound")
    #         ax.set_xlabel("Benders Iteration")
    #         ax.set_ylabel("Objective (£)")
    #         ax.set_title(f"Q2 Benders Convergence — {key}")
    #         ax.legend()
    #         ax.grid(True, alpha=0.3)
    #         fig.tight_layout()
    #         fig.savefig(
    #             str(_RESULTS_DIR / f"q2_benders_convergence_{key}.png"),
    #             dpi=150,
    #             bbox_inches="tight",
    #         )
    #         plt.close(fig)

    # ==================================================================
    # Save JSON
    # ==================================================================
    summary = {
        "method": "SAA + Benders",
        "small_instances": {},
        "full_instance": {},
        "cvar_sensitivity": {},
        "vss_evpi": {},
    }

    for label, r in small_results.items():
        if r["y"] is not None:
            summary["small_instances"][label] = {
                "objective": r["objective"],
                "gap": r["gap"],
                "iterations": r["iterations"],
                "time": r["time"],
                "open_stations": r.get("open_stations"),
                "total_chargers": r.get("total_chargers"),
                "cvar_095": r.get("cvar"),
            }

    # for key, r in all_results.items():
    #     if r["y"] is not None:
    #         summary["full_instance"][key] = {
    #             "objective": r["objective"],
    #             "gap": r["gap"],
    #             "time": r["time"],
    #             "open_stations": r.get("open_stations"),
    #             "total_chargers": r.get("total_chargers"),
    #             "cvar_095": r.get("cvar"),
    #         }

    summary["full_instance"]["eev"] = {
        "expected_cost": eev_result["expected_cost"],
        "first_stage": eev_result["first_stage_cost"],
        "second_stage": eev_result["expected_second_stage"],
        "cvar_095": cvar_eev,
        "var_095": var_eev,
    }

    for alpha, r in cvar_results.items():
        if r["y"] is not None:
            summary["cvar_sensitivity"][str(alpha)] = {
                "objective": r["objective"],
                "open_stations": r.get("open_stations"),
                "total_chargers": r.get("total_chargers"),
                "cvar_095": r.get("cvar"),
            }

    summary["vss_evpi"] = {
        "ws_50": ws_cost,
        "rp_50": rp_cost,
        "eev_50": eev_cost_50,
        "vss_50": vss,
        "evpi_50": evpi,
    }

    with open(_RESULTS_DIR / "q2_benders_full_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {_RESULTS_DIR / 'q2_benders_full_results.json'}")
    print(f"Log:   {_RESULTS_DIR / 'q2_benders_full_output.txt'}")

    tee.close()


if __name__ == "__main__":
    main()
