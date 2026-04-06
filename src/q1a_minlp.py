"""
Q1a: MINLP formulation for Antarctic charging station placement.

Formulates an exact Mixed-Integer Nonlinear Programme (MINLP) using the
Xpress solver. Station coordinates are continuous decision variables;
the nonlinear sqrt distance term makes this a MINLP.

Distance approximation:
    Azimuthal equidistant projection centred on the South Pole converts
    (lat, lon) to planar (x, y) in miles, avoiding the antimeridian
    discontinuity.  The resulting Euclidean distance in projected space is
    a good approximation for Antarctic coordinates.

Usage:
    python q1a_minlp.py   # from src/, or: python src/q1a_minlp.py from project root
"""

import numpy as np
import time
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, ".")

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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

import xpress as xp
xp.init("/Applications/FICO Xpress/xpressmp/bin/xpauth.xpr")

from data_loader import load_data, get_subset
from solution import (
    Solution, evaluate, check_feasibility, update_transport_flags,
)

EARTH_RADIUS_MILES = 3958.8


# ============================================================================
# Azimuthal equidistant projection (South Pole)
# ============================================================================

def to_polar(lat, lon):
    """Project (lat, lon) degrees → (x, y) miles (South-Pole centred)."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    r = EARTH_RADIUS_MILES * (np.pi / 2 + lat_r)   # lat < 0 for south
    return r * np.cos(lon_r), r * np.sin(lon_r)


def from_polar(px, py):
    """Inverse projection: (x, y) miles → (lat, lon) degrees."""
    r = np.sqrt(px ** 2 + py ** 2)
    lon = np.degrees(np.arctan2(py, px))
    lat = np.degrees(r / EARTH_RADIUS_MILES) - 90.0
    return lat, lon


# ============================================================================
# MINLP model construction
# ============================================================================

def build_model(data, J, time_limit=300):
    """Build the MINLP model for J candidate stations.

    Decision variables
    ------------------
    y_j  ∈ {0,1}         open station j
    nc_j ∈ {0,..,m}      chargers at station j
    sx_j, sy_j ∈ ℝ       station coordinates (projected space, miles)
    a_ij ∈ {0,1}         assign robot i to station j
    h_i  ∈ {0,1}         robot i needs human transport
    D_i  ∈ ℝ⁺            distance from robot i to its assigned station

    Objective (linear in D)
    -----------------------
    min  Σ cb·y_j + Σ cm·nc_j + cc·Σ D_i + ch·Σ h_i

    Constraints
    -----------
    C1  Σ_j a_ij = 1                          (assign each robot)
    C2  a_ij ≤ y_j                            (only open stations)
    C3  Σ_i a_ij ≤ nc_j · q                   (station capacity)
    C4  nc_j ≤ m · y_j                        (charger limit)
    C5  D_i ≥ √((rx_i-sx_j)²+(ry_i-sy_j)²+ε) − M(1−a_ij)
                                               (distance linking, NL)
    C6  D_i ≤ range_i + M · h_i               (transport flag)
    C7  sx_j ≤ sx_{j+1}                       (symmetry breaking)

    Returns
    -------
    prob : xpress.problem
    vd   : dict of variable references
    """
    n = data.n_robots

    # ── project robot coordinates ───────────────────────────────────────
    rx, ry = to_polar(data.robot_lat, data.robot_lon)

    pad = 200.0
    x_lo, x_hi = float(rx.min() - pad), float(rx.max() + pad)
    y_lo, y_hi = float(ry.min() - pad), float(ry.max() + pad)

    M = float(np.sqrt((x_hi - x_lo) ** 2 + (y_hi - y_lo) ** 2)) + 500

    print(f"  Projected: x∈[{rx.min():.0f}, {rx.max():.0f}], "
          f"y∈[{ry.min():.0f}, {ry.max():.0f}] miles")
    print(f"  Big-M = {M:.0f} miles")

    prob = xp.problem("Q1a_MINLP")

    # ── decision variables ──────────────────────────────────────────────
    y_var = [prob.addVariable(vartype=xp.binary, name=f"y{j}")
             for j in range(J)]
    nc = [prob.addVariable(vartype=xp.integer, lb=0, ub=data.m,
                           name=f"nc{j}")
          for j in range(J)]
    sx = [prob.addVariable(lb=x_lo, ub=x_hi, name=f"sx{j}")
          for j in range(J)]
    sy = [prob.addVariable(lb=y_lo, ub=y_hi, name=f"sy{j}")
          for j in range(J)]

    a = [[prob.addVariable(vartype=xp.binary, name=f"a{i}_{j}")
          for j in range(J)] for i in range(n)]
    h = [prob.addVariable(vartype=xp.binary, name=f"h{i}")
         for i in range(n)]
    D = [prob.addVariable(lb=0, ub=M, name=f"D{i}")
         for i in range(n)]

    # ── objective (linear in D) ───────────────────────────────────────
    # cc × D[i] only for self-flying robots: cc * D[i] * (1 - h[i])
    obj = (xp.Sum(data.cb * y_var[j] + data.cm * nc[j] for j in range(J))
           + data.cc * xp.Sum(D[i] * (1 - h[i]) for i in range(n))
           + data.ch * xp.Sum(h[i] for i in range(n)))
    prob.setObjective(obj, sense=xp.minimize)

    # ── constraints ─────────────────────────────────────────────────────
    # C1: each robot assigned to exactly one station
    for i in range(n):
        prob.addConstraint(xp.Sum(a[i][j] for j in range(J)) == 1)

    # C2: assignment only to open stations
    for i in range(n):
        for j in range(J):
            prob.addConstraint(a[i][j] <= y_var[j])

    # C3: station capacity
    for j in range(J):
        prob.addConstraint(
            xp.Sum(a[i][j] for i in range(n)) <= nc[j] * data.q)

    # C4: charger limit
    for j in range(J):
        prob.addConstraint(nc[j] <= data.m * y_var[j])

    # C5: distance linking (NONLINEAR)
    EPS = 1e-4  # numerical stability inside sqrt
    for i in range(n):
        rxi, ryi = float(rx[i]), float(ry[i])
        for j in range(J):
            dx = rxi - sx[j]
            dy = ryi - sy[j]
            prob.addConstraint(
                D[i] >= xp.sqrt(dx * dx + dy * dy + EPS)
                - M * (1 - a[i][j]))

    # C6: transport flag
    for i in range(n):
        prob.addConstraint(
            D[i] <= float(data.robot_range[i]) + M * h[i])

    # C7: symmetry breaking (order stations by x-coordinate)
    for j in range(J - 1):
        prob.addConstraint(sx[j] <= sx[j + 1])

    # ── solver controls ─────────────────────────────────────────────────
    prob.controls.maxtime = time_limit
    prob.controls.timelimit = time_limit  # global/NL solver time limit
    prob.controls.outputlog = 0           # suppress verbose solver output

    n_bin = J + n * J + n
    n_nl = n * J
    print(f"  Variables: {J * 4 + n * J + n * 2} "
          f"({n_bin} binary, {J} integer, {J * 2 + n} continuous)")
    print(f"  Constraints: ~{n + n * J + J * 2 + n * J + n + J - 1} "
          f"({n_nl} nonlinear)")

    vd = {"y": y_var, "nc": nc, "sx": sx, "sy": sy,
          "a": a, "h": h, "D": D}
    return prob, vd


# ============================================================================
# Solution extraction
# ============================================================================

def extract_solution(prob, vd, data):
    """Convert solved Xpress variables → Solution object."""
    n = data.n_robots
    J = len(vd["y"])

    y_val = np.array([prob.getSolution(vd["y"][j]) for j in range(J)])
    nc_val = np.array([prob.getSolution(vd["nc"][j]) for j in range(J)])
    sx_val = np.array([prob.getSolution(vd["sx"][j]) for j in range(J)])
    sy_val = np.array([prob.getSolution(vd["sy"][j]) for j in range(J)])
    h_val = np.array([prob.getSolution(vd["h"][i]) for i in range(n)])

    a_val = np.zeros((n, J))
    for i in range(n):
        for j in range(J):
            a_val[i, j] = prob.getSolution(vd["a"][i][j])

    assignments = np.argmax(a_val, axis=1)

    # Keep only open stations
    open_mask = y_val > 0.5
    open_idx = np.where(open_mask)[0]
    remap = np.full(J, -1, dtype=int)
    remap[open_idx] = np.arange(len(open_idx))

    # Convert projected coords → lat/lon
    s_lat, s_lon = from_polar(sx_val[open_mask], sy_val[open_mask])

    return Solution(
        station_lon=s_lon,
        station_lat=s_lat,
        n_chargers=np.round(nc_val[open_mask]).astype(int),
        assignments=remap[assignments],
        needs_transport=(h_val > 0.5),
    )


# ============================================================================
# Solve wrapper
# ============================================================================

def solve_instance(data, J, time_limit=300, label=""):
    """Build, solve, and report results for one instance.

    Returns dict with keys: sol, cost, minlp_obj, time, status, gap.
    """
    print(f"\n{'=' * 60}")
    print(f"MINLP Instance: {label}")
    print(f"  n={data.n_robots}, J={J}, time_limit={time_limit}s")
    print(f"{'=' * 60}")

    # Build
    t0 = time.time()
    prob, vd = build_model(data, J, time_limit)
    t_build = time.time() - t0
    print(f"  Model built in {t_build:.1f}s")

    # Solve
    t0 = time.time()
    prob.solve()
    t_solve = time.time() - t0

    result = {
        "status": "unknown", "time": t_solve, "build_time": t_build,
        "sol": None, "cost": None, "minlp_obj": None, "gap": None,
    }

    # Check status
    try:
        mip_status = prob.attributes.mipstatus
        status_map = {
            0: "not started", 1: "LP not solved", 2: "LP optimal",
            3: "search incomplete", 4: "solution found", 5: "no solution",
            6: "optimal", 7: "cutoff",
        }
        result["status"] = status_map.get(mip_status, f"mip={mip_status}")
    except Exception:
        try:
            sol_status = prob.attributes.solstatus
            result["status"] = f"sol={sol_status}"
        except Exception:
            pass

    print(f"\n  Status:     {result['status']}")
    print(f"  Solve time: {t_solve:.1f}s")

    # Try to get objective
    try:
        obj_val = prob.attributes.mipobjval
        result["minlp_obj"] = obj_val
        print(f"  MINLP obj:  £{obj_val:,.2f}")
    except Exception:
        try:
            obj_val = prob.attributes.objval
            result["minlp_obj"] = obj_val
            print(f"  Obj val:    £{obj_val:,.2f}")
        except Exception:
            print("  No objective value available")
            return result

    # Compute gap from best bound
    try:
        best_bound = prob.attributes.bestbound
        if result["minlp_obj"] and best_bound and result["minlp_obj"] > 0:
            gap = (result["minlp_obj"] - best_bound) / result["minlp_obj"]
            result["gap"] = gap
            print(f"  Best bound: £{best_bound:,.2f}")
            print(f"  Gap:        {gap:.4f} ({gap * 100:.2f}%)")
    except Exception:
        pass

    # Extract solution
    try:
        sol = extract_solution(prob, vd, data)
    except Exception as e:
        print(f"  Solution extraction failed: {e}")
        return result

    # Evaluate with Haversine (true cost)
    update_transport_flags(sol, data)
    cost = evaluate(sol, data)

    result["sol"] = sol
    result["cost"] = cost

    print(f"\n  Solution: {sol.n_stations} stations opened")
    print(f"  Chargers: {sol.n_chargers.tolist()}")
    print(f"  Transport needed: {int(np.sum(sol.needs_transport))} robots")
    print(f"\n  Haversine-based cost:")
    print(f"    Build:       £{cost.build:,.2f}")
    print(f"    Maintenance: £{cost.maintenance:,.2f}")
    print(f"    Charging:    £{cost.charging:,.2f}")
    print(f"    Transport:   £{cost.transport:,.2f}")
    print(f"    TOTAL:       £{cost.total:,.2f}")

    feasible, violations = check_feasibility(sol, data)
    print(f"  Feasible (Haversine): {feasible}")
    if not feasible:
        for v in violations:
            print(f"    - {v}")

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = str(_RESULTS_DIR / "q1a_output.txt")
    tee = Tee(log_path)
    sys.stdout = tee

    print("Q1a: MINLP Formulation for Charging Station Placement")
    print("=" * 60)

    data = load_data()
    print(f"Full data loaded: {data.n_robots} robots\n")

    rng = np.random.default_rng(42)

    # Instance configurations: (n_robots, J_upper_bound, time_limit_s)
    instances = [
        (20,  5,  1800),
        (50,  8,  1800),
        (100, 15, 1800),
    ]

    all_results = {}

    for n_robots, J, tlim in instances:
        idx = rng.choice(data.n_robots, n_robots, replace=False)
        sub = get_subset(data, idx)
        label = f"{n_robots} robots"

        r = solve_instance(sub, J, time_limit=tlim, label=label)
        all_results[label] = r

        # Save visualisation
        if r["sol"] is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from visualization import plot_solution_map, plot_cost_breakdown

            fig, _ = plot_solution_map(
                r["sol"], sub,
                title=f"Q1a MINLP — {label}",
                save_path=str(_RESULTS_DIR / f"q1a_map_{n_robots}.png"))
            plt.close(fig)

            fig, _ = plot_cost_breakdown(
                r["cost"],
                title=f"Q1a Cost Breakdown — {label}",
                save_path=str(_RESULTS_DIR / f"q1a_cost_{n_robots}.png"))
            plt.close(fig)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print("Q1a MINLP Summary")
    print(f"{'=' * 75}")
    print(f"{'Instance':<15} {'Status':<20} {'Stations':>8} "
          f"{'MINLP Obj':>12} {'Haversine':>12} {'Time':>8}")
    print("-" * 75)

    for label, r in all_results.items():
        ns = str(r["sol"].n_stations) if r["sol"] else "—"
        mo = f"£{r['minlp_obj']:,.0f}" if r["minlp_obj"] else "—"
        hc = f"£{r['cost'].total:,.0f}" if r["cost"] else "—"
        gap_s = f" (gap={r['gap']:.4f})" if r["gap"] else ""
        print(f"{label:<15} {r['status'] + gap_s:<20} {ns:>8} "
              f"{mo:>12} {hc:>12} {r['time']:>7.1f}s")

    # ── Save JSON ───────────────────────────────────────────────────────
    summary = {}
    for label, r in all_results.items():
        entry = {
            "status": r["status"], "time": r["time"],
            "build_time": r["build_time"],
        }
        if r["cost"]:
            entry["haversine_cost"] = r["cost"].as_dict()
        if r["minlp_obj"] is not None:
            entry["minlp_obj"] = r["minlp_obj"]
        if r["gap"] is not None:
            entry["gap"] = r["gap"]
        if r["sol"]:
            entry["n_stations"] = r["sol"].n_stations
            entry["n_chargers"] = r["sol"].n_chargers.tolist()
            entry["transport_count"] = int(np.sum(r["sol"].needs_transport))
            entry["station_lon"] = r["sol"].station_lon.tolist()
            entry["station_lat"] = r["sol"].station_lat.tolist()
        summary[label] = entry

    with open(_RESULTS_DIR / "q1a_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {_RESULTS_DIR / 'q1a_results.json'}")
    print(f"Output log saved to {_RESULTS_DIR / 'q1a_output.txt'}")

    tee.close()


if __name__ == "__main__":
    main()
