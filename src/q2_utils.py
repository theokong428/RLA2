"""
Q2 shared utility functions for stochastic evaluation.

Used by q2_salns.py (Stochastic ALNS) and q2_benders.py / q2_benders_full.py.
No Xpress dependency.
"""

import numpy as np
from utils import distance_matrix, charging_probability


def evaluate_stochastic(data, sol):
    """Evaluate a Q1 deterministic solution under all stochastic scenarios.

    For each scenario s:
      - Re-optimise assignment greedily (nearest feasible station)
      - Compute cost weighted by charging probability p_i^s

    Returns:
        expected_cost: first_stage + (1/S) sum scenario_cost_s
        scenario_costs: (n_scenarios,) per-scenario second-stage costs
        first_stage_cost: cb * n_stations + cm * sum n_chargers
    """
    n_s = data.n_scenarios
    n_i = data.n_robots

    first_stage = data.cb * sol.n_stations + data.cm * float(
        np.sum(sol.n_chargers))

    dist_mat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )
    cap = sol.n_chargers * data.q

    scenario_costs = np.zeros(n_s)

    for s in range(n_s):
        ranges_s = data.scenario_range[:, s]
        p_s = charging_probability(ranges_s, data.lam, data.r_min)

        # Greedy nearest-feasible assignment for this scenario
        counts = np.zeros(sol.n_stations, dtype=int)
        total_cost = 0.0

        order = np.argsort(dist_mat.min(axis=1))
        for i in order:
            assigned = False
            for j in np.argsort(dist_mat[i]):
                if counts[j] < cap[j]:
                    d = dist_mat[i, j]
                    transport = 1.0 if d > ranges_s[i] else 0.0
                    fly = 1.0 - transport
                    total_cost += p_s[i] * (
                        data.cc * d * fly + data.ch * transport)
                    counts[j] += 1
                    assigned = True
                    break
            if not assigned:
                # Assign to nearest station anyway (overflow)
                j = np.argmin(dist_mat[i])
                d = dist_mat[i, j]
                transport = 1.0 if d > ranges_s[i] else 0.0
                fly = 1.0 - transport
                total_cost += p_s[i] * (data.cc * d * fly + data.ch * transport)
                counts[j] += 1

        scenario_costs[s] = total_cost

    expected_second = float(np.mean(scenario_costs))
    return {
        "expected_cost": first_stage + expected_second,
        "first_stage_cost": first_stage,
        "expected_second_stage": expected_second,
        "scenario_costs": scenario_costs,
    }


def compute_cvar(scenario_costs, beta=0.95):
    """Compute CVaR_beta of scenario costs.

    CVaR_beta = E[cost | cost >= VaR_beta]

    Returns (cvar, var).
    """
    sorted_c = np.sort(scenario_costs)
    idx = int(np.floor(len(sorted_c) * beta))
    var = sorted_c[min(idx, len(sorted_c) - 1)]
    cvar = float(np.mean(sorted_c[idx:]))
    return cvar, var
