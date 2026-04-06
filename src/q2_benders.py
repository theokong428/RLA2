"""
Q2: Benders decomposition engine for two-stage stochastic CFLP.

Implements the L-shaped method for a capacitated facility location problem
with scenario-dependent robot ranges and charging probabilities.

Master problem (MIP):
    First-stage: y_j (open station), n_j (chargers), fixed candidate locations
    Per-scenario cost approximation: θ_s
    Optional Mean-CVaR objective (Nazemi et al., 2022)

Subproblem (LP per scenario s):
    Second-stage: x_ij assignment, weighted by charging probability p_i^s
    Cost: p_i^s * (cc * d_ij + ch * I(d_ij > r_i^s))

References:
    Parragh et al. (2022), OR Spectrum — Benders for stochastic FL
    Rockafellar & Uryasev (2000) — CVaR linearisation
"""

import numpy as np
import time

import xpress as xp
xp.init("/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr")

from data_loader import ProblemData
from utils import distance_matrix, charging_probability


# ============================================================================
# Precompute scenario cost coefficients
# ============================================================================

def precompute_scenario_costs(data, cand_lon, cand_lat):
    """Compute per-scenario assignment cost coefficients.

    c_ij^s = p_i^s * (cc * d_ij + ch * I(d_ij > r_i^s))

    Args:
        data:     ProblemData with scenario_range.
        cand_lon: (n_cand,) candidate station longitudes.
        cand_lat: (n_cand,) candidate station latitudes.

    Returns:
        cost_coeffs: (n_robots, n_cand, n_scenarios) array.
        dist_mat:    (n_robots, n_cand) distance matrix.
    """
    n_i = data.n_robots
    n_j = len(cand_lon)
    n_s = data.n_scenarios

    dist_mat = distance_matrix(
        data.robot_lat, data.robot_lon,
        cand_lat, cand_lon,
    )

    cost_coeffs = np.zeros((n_i, n_j, n_s))
    for s in range(n_s):
        ranges_s = data.scenario_range[:, s]
        p_s = charging_probability(ranges_s, data.lam, data.r_min)
        for j in range(n_j):
            transport_flag = (dist_mat[:, j] > ranges_s).astype(float)
            fly_flag = 1.0 - transport_flag
            cost_coeffs[:, j, s] = p_s * (
                data.cc * dist_mat[:, j] * fly_flag + data.ch * transport_flag
            )

    return cost_coeffs, dist_mat


# ============================================================================
# Benders decomposition
# ============================================================================

class BendersDecomposition:
    """L-shaped Benders decomposition for two-stage stochastic CFLP.

    Candidate station locations are fixed (discrete set). The master
    optimises open/charger decisions; subproblems solve per-scenario
    assignment LPs.
    """

    def __init__(self, data, cand_lon, cand_lat,
                 alpha_cvar=0.0, beta_cvar=0.95, time_limit=600):
        self.data = data
        self.n_robots = data.n_robots
        self.n_cand = len(cand_lon)
        self.cand_lon = np.asarray(cand_lon)
        self.cand_lat = np.asarray(cand_lat)
        self.n_scenarios = data.n_scenarios
        self.alpha_cvar = alpha_cvar
        self.beta_cvar = beta_cvar
        self.time_limit = time_limit

        # Precompute
        self.cost_coeffs, self.dist_mat = precompute_scenario_costs(
            data, cand_lon, cand_lat)

        # Will be set by _build_master
        self.master = None
        self.y_vars = None
        self.n_vars = None
        self.theta_vars = None
        self.eta_var = None
        self.z_vars = None

    # ── Master problem ─────────────────────────────────────────────────

    def _build_master(self):
        prob = xp.problem("Benders_Master")
        prob.controls.outputlog = 0
        prob.controls.maxtime = self.time_limit
        prob.controls.timelimit = self.time_limit

        nj = self.n_cand
        ns = self.n_scenarios
        m = self.data.m
        q = self.data.q

        # First-stage variables
        y = [prob.addVariable(vartype=xp.binary, name=f"y{j}")
             for j in range(nj)]
        n = [prob.addVariable(vartype=xp.integer, lb=0, ub=m, name=f"n{j}")
             for j in range(nj)]

        # Per-scenario second-stage cost approximation
        theta = [prob.addVariable(lb=0, name=f"th{s}") for s in range(ns)]

        # Linking: n_j ≤ m * y_j
        for j in range(nj):
            prob.addConstraint(n[j] <= m * y[j])

        # Total capacity ≥ n_robots
        prob.addConstraint(
            xp.Sum(n[j] * q for j in range(nj)) >= self.n_robots)

        # Objective
        first_stage = xp.Sum(
            self.data.cb * y[j] + self.data.cm * n[j] for j in range(nj))

        alpha = self.alpha_cvar
        if alpha > 1e-9:
            # Mean-CVaR objective
            eta = prob.addVariable(lb=-1e8, name="eta")
            z = [prob.addVariable(lb=0, name=f"z{s}") for s in range(ns)]
            for s in range(ns):
                prob.addConstraint(z[s] >= theta[s] - eta)

            risk_neutral_part = (1 - alpha) / ns * xp.Sum(
                theta[s] for s in range(ns))
            cvar_part = alpha * (
                eta + 1.0 / (ns * (1 - self.beta_cvar))
                * xp.Sum(z[s] for s in range(ns)))

            prob.setObjective(
                first_stage + risk_neutral_part + cvar_part,
                sense=xp.minimize)
            self.eta_var = eta
            self.z_vars = z
        else:
            # Pure risk-neutral
            prob.setObjective(
                first_stage + 1.0 / ns * xp.Sum(
                    theta[s] for s in range(ns)),
                sense=xp.minimize)

        self.master = prob
        self.y_vars = y
        self.n_vars = n
        self.theta_vars = theta

    # ── Subproblem ─────────────────────────────────────────────────────

    def _solve_subproblem(self, s, y_vals, n_vals):
        """Solve the LP subproblem for scenario s.

        Returns (obj, alpha_dual, beta_dual, feasible).
        alpha_dual: (n_robots,) duals of assignment constraints.
        beta_dual:  (n_cand,) duals of capacity constraints.
        """
        ni = self.n_robots
        nj = self.n_cand
        q = self.data.q
        PENALTY = 1e5  # slack penalty for infeasibility

        # Capacity per station
        cap = np.array([n_vals[j] * q for j in range(nj)])

        prob = xp.problem(f"Sub{s}")
        prob.controls.outputlog = 0

        # Variables: x_ij ∈ [0,1], plus slack_i ≥ 0
        x_flat = [prob.addVariable(lb=0, ub=1.0, name=f"x{i}_{j}")
                  for i in range(ni) for j in range(nj)]
        slack = [prob.addVariable(lb=0, name=f"sl{i}")
                 for i in range(ni)]

        # C1: Σ_j x_ij + slack_i = 1  (assignment, rows 0..ni-1)
        for i in range(ni):
            prob.addConstraint(
                xp.Sum(x_flat[i * nj + j] for j in range(nj))
                + slack[i] == 1)

        # C2: Σ_i x_ij ≤ cap_j  (capacity, rows ni..ni+nj-1)
        for j in range(nj):
            prob.addConstraint(
                xp.Sum(x_flat[i * nj + j] for i in range(ni)) <= cap[j])

        # Objective
        obj_expr = xp.Sum(
            self.cost_coeffs[i, j, s] * x_flat[i * nj + j]
            for i in range(ni) for j in range(nj)
        ) + PENALTY * xp.Sum(slack[i] for i in range(ni))

        prob.setObjective(obj_expr, sense=xp.minimize)
        prob.solve()

        status = prob.attributes.solstatus
        if status not in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
            return None, None, None, False

        obj_val = prob.attributes.objval

        # Extract duals via getlpsol (Xpress 9.7: pass empty lists)
        x_arr = []
        sl_arr = []
        dual_arr = []
        dj_arr = []
        prob.getlpsol(x_arr, sl_arr, dual_arr, dj_arr)

        dual_arr = np.array(dual_arr)
        alpha_dual = dual_arr[:ni]          # assignment duals
        beta_dual = dual_arr[ni:ni + nj]    # capacity duals

        return obj_val, alpha_dual, beta_dual, True

    # ── Cut generation ─────────────────────────────────────────────────

    def _add_optimality_cut(self, s, alpha_dual, beta_dual):
        """Add Benders optimality cut: θ_s ≥ Σ α_i + Σ β_j * n_j * q."""
        q = self.data.q
        rhs_const = float(np.sum(alpha_dual))

        self.master.addConstraint(
            self.theta_vars[s] >= rhs_const + xp.Sum(
                float(beta_dual[j]) * q * self.n_vars[j]
                for j in range(self.n_cand)
            )
        )

    # ── Main loop ──────────────────────────────────────────────────────

    def solve(self, max_iter=80, gap_tol=1e-3, verbose=True):
        """Run Benders decomposition.

        Returns dict with y, n, objective, gap, scenario_costs, history.
        """
        self._build_master()

        UB = np.inf
        LB = -np.inf
        best_y, best_n = None, None
        best_obj = np.inf
        history = []
        scenario_costs = np.zeros(self.n_scenarios)

        t0 = time.time()

        for it in range(max_iter):
            # 1. Solve master
            self.master.solve()

            sol_status = self.master.attributes.solstatus
            if sol_status not in (
                    xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
                if verbose:
                    print(f"  Benders iter {it}: master solstatus="
                          f"{sol_status}, no solution")
                break

            try:
                master_obj = self.master.attributes.mipobjval
            except Exception:
                try:
                    master_obj = self.master.attributes.objval
                except Exception:
                    if verbose:
                        print(f"  Benders iter {it}: master no obj value")
                    break

            LB = max(LB, master_obj)

            y_vals = [round(self.master.getSolution(self.y_vars[j]))
                      for j in range(self.n_cand)]
            n_vals = [round(self.master.getSolution(self.n_vars[j]))
                      for j in range(self.n_cand)]

            # Read theta values (may be 0 in first iteration)
            theta_vals = []
            for s in range(self.n_scenarios):
                try:
                    theta_vals.append(
                        self.master.getSolution(self.theta_vars[s]))
                except Exception:
                    theta_vals.append(0.0)

            # Restore master to original form so we can add cuts
            self.master.postsolve()

            # 2. Solve all scenario subproblems
            total_sub = 0.0
            cuts_added = 0
            all_ok = True

            for s in range(self.n_scenarios):
                obj_s, a_d, b_d, ok = self._solve_subproblem(
                    s, y_vals, n_vals)

                if not ok:
                    all_ok = False
                    break

                scenario_costs[s] = obj_s
                total_sub += obj_s

                # Check if cut is violated
                if obj_s > theta_vals[s] + 1e-4:
                    self._add_optimality_cut(s, a_d, b_d)
                    cuts_added += 1

            if not all_ok:
                if verbose:
                    print(f"  Benders iter {it}: subproblem infeasible, "
                          "adding capacity cut")
                self.master.addConstraint(
                    xp.Sum(self.n_vars[j] * self.data.q
                           for j in range(self.n_cand))
                    >= self.n_robots + 16)
                continue

            # 3. Compute actual upper bound
            first_stage_cost = sum(
                self.data.cb * y_vals[j] + self.data.cm * n_vals[j]
                for j in range(self.n_cand))

            if self.alpha_cvar > 1e-9:
                rn = (1 - self.alpha_cvar) / self.n_scenarios * total_sub
                tail_start = int(
                    np.floor(self.n_scenarios * self.beta_cvar))
                sorted_sc = np.sort(scenario_costs)
                cvar_val = np.mean(sorted_sc[tail_start:])
                actual_obj = first_stage_cost + rn + self.alpha_cvar * cvar_val
            else:
                actual_obj = first_stage_cost + total_sub / self.n_scenarios

            UB = min(UB, actual_obj)
            if actual_obj < best_obj:
                best_obj = actual_obj
                best_y = y_vals[:]
                best_n = n_vals[:]

            gap = (UB - LB) / max(abs(UB), 1e-10) if UB < np.inf else 1.0
            elapsed = time.time() - t0
            history.append({
                "iter": it, "LB": LB, "UB": UB,
                "gap": gap, "cuts": cuts_added, "time": elapsed,
            })

            if verbose:
                print(f"  Benders {it:>3}: LB=£{LB:,.0f}  UB=£{UB:,.0f}  "
                      f"gap={gap:.4f}  cuts={cuts_added}  "
                      f"t={elapsed:.1f}s")

            if gap < gap_tol or cuts_added == 0:
                if verbose:
                    print(f"  Converged (gap={gap:.6f})")
                break

            if elapsed > self.time_limit:
                if verbose:
                    print(f"  Time limit reached ({elapsed:.0f}s)")
                break

        return {
            "y": best_y, "n": best_n,
            "objective": best_obj,
            "LB": LB, "UB": UB,
            "gap": (UB - LB) / max(abs(UB), 1e-10) if UB < np.inf else 1.0,
            "iterations": len(history),
            "history": history,
            "scenario_costs": scenario_costs.copy(),
            "time": time.time() - t0,
            "first_stage_cost": sum(
                self.data.cb * (best_y[j] if best_y else 0)
                + self.data.cm * (best_n[j] if best_n else 0)
                for j in range(self.n_cand)),
        }


# ============================================================================
# Stochastic evaluation of a deterministic solution
# ============================================================================

def evaluate_stochastic(data, sol):
    """Evaluate a Q1 deterministic solution under all stochastic scenarios.

    For each scenario s:
      - Re-optimise assignment greedily (nearest feasible station)
      - Compute cost weighted by charging probability p_i^s

    Returns:
        expected_cost: first_stage + (1/S) Σ scenario_cost_s
        scenario_costs: (n_scenarios,) per-scenario second-stage costs
        first_stage_cost: cb * n_stations + cm * Σ n_chargers
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

    CVaR_beta = E[cost | cost ≥ VaR_beta]

    Returns (cvar, var).
    """
    sorted_c = np.sort(scenario_costs)
    idx = int(np.floor(len(sorted_c) * beta))
    var = sorted_c[min(idx, len(sorted_c) - 1)]
    cvar = float(np.mean(sorted_c[idx:]))
    return cvar, var
