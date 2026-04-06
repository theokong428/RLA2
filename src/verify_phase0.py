"""
Phase 0 verification: load data, test core functions, compare with Data_Xpress.txt.
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from data_loader import load_data, get_subset
from utils import haversine, distance_matrix, charging_probability
from solution import Solution, evaluate, check_feasibility, update_transport_flags, update_charger_counts


def main():
    print("=" * 60)
    print("Phase 0 Verification")
    print("=" * 60)

    # --- 1. Load data ---
    data = load_data()
    print(f"\n[1] Data loaded:")
    print(f"    n_robots     = {data.n_robots}")
    print(f"    n_scenarios  = {data.n_scenarios}")
    print(f"    m={data.m}, q={data.q}, cb={data.cb}, ch={data.ch}, "
          f"cm={data.cm}, cc={data.cc}")
    print(f"    λ={data.lam}, r_min={data.r_min}, r_max={data.r_max}")
    assert data.n_robots == 1072, f"Expected 1072 robots, got {data.n_robots}"
    assert data.n_scenarios == 100, f"Expected 100 scenarios, got {data.n_scenarios}"
    assert data.scenario_range.shape == (1072, 100)

    # --- 2. Cross-check with Data_Xpress.txt ---
    print(f"\n[2] Cross-check first 3 robots with Data_Xpress.txt:")
    print(f"    Robot 0: lon={data.robot_lon[0]:.7f}, lat={data.robot_lat[0]:.8f}")
    print(f"             range={data.robot_range[0]:.7f}")
    print(f"    Robot 1: lon={data.robot_lon[1]:.7f}, lat={data.robot_lat[1]:.7f}")
    print(f"             range={data.robot_range[1]:.7f}")
    # Expected from Data_Xpress.txt:
    # Robot 0: lat=-79.23837427, lon=-120.2408148, range=127.8682745
    assert abs(data.robot_lat[0] - (-79.23837427)) < 1e-5
    assert abs(data.robot_lon[0] - (-120.2408148)) < 1e-5
    assert abs(data.robot_range[0] - 127.8682745) < 1e-3
    print("    ✓ Data matches Data_Xpress.txt")

    # --- 3. Range statistics ---
    print(f"\n[3] Deterministic range statistics:")
    print(f"    min  = {data.robot_range.min():.2f} miles")
    print(f"    max  = {data.robot_range.max():.2f} miles")
    print(f"    mean = {data.robot_range.mean():.2f} miles")
    print(f"    std  = {data.robot_range.std():.2f} miles")

    # --- 4. Coordinate bounds ---
    print(f"\n[4] Coordinate bounds:")
    print(f"    Longitude: [{data.robot_lon.min():.2f}, {data.robot_lon.max():.2f}]")
    print(f"    Latitude:  [{data.robot_lat.min():.2f}, {data.robot_lat.max():.2f}]")

    # --- 5. Haversine test ---
    print(f"\n[5] Haversine sanity check:")
    # Two robots far apart: 0 (lon=-120, lat=-79) and 1 (lon=-134, lat=-83)
    d01 = haversine(data.robot_lat[0], data.robot_lon[0],
                    data.robot_lat[1], data.robot_lon[1])
    print(f"    d(robot 0, robot 1) = {d01:.2f} miles")
    assert d01 > 0 and d01 < 5000, f"Unreasonable distance: {d01}"

    # Antipodal-like check: two points at south pole
    d_pole = haversine(-90, 0, -90, 180)
    print(f"    d(south pole, south pole via 180°) = {d_pole:.2f} miles (should be ~0)")
    assert d_pole < 1.0, f"South pole distance should be ~0, got {d_pole}"
    print("    ✓ Haversine working correctly")

    # --- 6. Distance matrix test ---
    print(f"\n[6] Distance matrix test (5 robots × 2 dummy stations):")
    sub = get_subset(data, np.arange(5))
    s_lon = np.array([-120.0, -134.0])
    s_lat = np.array([-80.0, -83.0])
    dmat = distance_matrix(sub.robot_lat, sub.robot_lon, s_lat, s_lon)
    print(f"    Shape: {dmat.shape}")
    print(f"    Distances:\n{dmat.round(2)}")
    assert dmat.shape == (5, 2)
    assert np.all(dmat >= 0)
    print("    ✓ Distance matrix OK")

    # --- 7. Charging probability test ---
    print(f"\n[7] Charging probability:")
    test_ranges = np.array([10, 50, 100, 150, 175])
    probs = charging_probability(test_ranges, lam=data.lam, r_min=data.r_min)
    for r, p in zip(test_ranges, probs):
        print(f"    range={r:3d} miles → p={p:.6f}")
    assert abs(probs[0] - 1.0) < 1e-10, "p(r_min) should be 1.0"
    assert probs[-1] < probs[0], "p should decrease with range"
    print("    ✓ Charging probability correct")

    # --- 8. Solution evaluate & feasibility test ---
    print(f"\n[8] Solution evaluation (5-robot toy example):")
    sub5 = get_subset(data, np.arange(5))
    toy_sol = Solution(
        station_lon=np.array([-115.0, -140.0]),
        station_lat=np.array([-80.0, -84.0]),
        n_chargers=np.array([4, 4]),
        assignments=np.array([0, 1, 0, 0, 1]),
        needs_transport=np.zeros(5, dtype=bool),
    )
    # Update transport flags based on actual distances
    update_transport_flags(toy_sol, sub5)
    cost = evaluate(toy_sol, sub5)
    print(f"    Build:       £{cost.build:,.2f}")
    print(f"    Maintenance: £{cost.maintenance:,.2f}")
    print(f"    Charging:    £{cost.charging:,.2f}")
    print(f"    Transport:   £{cost.transport:,.2f}")
    print(f"    TOTAL:       £{cost.total:,.2f}")
    print(f"    Needs transport: {toy_sol.needs_transport}")

    feasible, violations = check_feasibility(toy_sol, sub5)
    print(f"    Feasible: {feasible}")
    if not feasible:
        for v in violations:
            print(f"      - {v}")

    # --- 9. Subset extraction test ---
    print(f"\n[9] Subset extraction:")
    idx = np.random.default_rng(42).choice(data.n_robots, 50, replace=False)
    sub50 = get_subset(data, idx)
    print(f"    50-robot subset: n_robots={sub50.n_robots}, "
          f"scenario_range shape={sub50.scenario_range.shape}")
    assert sub50.n_robots == 50
    assert sub50.scenario_range.shape == (50, 100)
    print("    ✓ Subset extraction OK")

    print("\n" + "=" * 60)
    print("All Phase 0 checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
