"""
Core solution representation, evaluation, and feasibility checking.

Shared by Q1b (construction), Q1c (SA / ALNS improvement), and Q2 (stochastic).
"""

import numpy as np
from dataclasses import dataclass
from data_loader import ProblemData
from utils import distance_matrix


# ---------------------------------------------------------------------------
# Solution data structure
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    """A charging-station placement solution.

    Attributes:
        station_lon:     (n_stations,) longitude of each station.
        station_lat:     (n_stations,) latitude of each station.
        n_chargers:      (n_stations,) number of chargers at each station.
        assignments:     (n_robots,) station index each robot is assigned to.
        needs_transport: (n_robots,) True if robot requires human transport.
    """
    station_lon: np.ndarray
    station_lat: np.ndarray
    n_chargers: np.ndarray
    assignments: np.ndarray
    needs_transport: np.ndarray

    @property
    def n_stations(self) -> int:
        return len(self.station_lon)

    def copy(self) -> "Solution":
        """Deep copy."""
        return Solution(
            station_lon=self.station_lon.copy(),
            station_lat=self.station_lat.copy(),
            n_chargers=self.n_chargers.copy(),
            assignments=self.assignments.copy(),
            needs_transport=self.needs_transport.copy(),
        )


# ---------------------------------------------------------------------------
# Cost breakdown
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    """Itemised cost breakdown."""
    build: float        # cb × number of open stations
    maintenance: float  # cm × total chargers
    charging: float     # cc × Σ d_ij × (1 - h_i), self-flying only
    transport: float    # ch × number of transported robots

    @property
    def total(self) -> float:
        return self.build + self.maintenance + self.charging + self.transport

    def as_dict(self) -> dict:
        return {
            "build": self.build,
            "maintenance": self.maintenance,
            "charging": self.charging,
            "transport": self.transport,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(sol: Solution, data: ProblemData) -> CostBreakdown:
    """Compute the deterministic total cost of a solution (Q1).

    Cost = cb·|stations| + cm·Σn_j + cc·Σ[d_ij·(1-h_i)] + ch·Σh_i

    Args:
        sol:  Solution to evaluate.
        data: Problem data.

    Returns:
        CostBreakdown with all components.
    """
    build = data.cb * sol.n_stations
    maintenance = data.cm * float(np.sum(sol.n_chargers))

    # Robot-to-assigned-station distances (vectorised)
    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )
    robot_dists = dmat[np.arange(data.n_robots), sol.assignments]

    # cc × d_ij only for self-flying robots (needs_transport=False)
    fly_dists = robot_dists * (~sol.needs_transport).astype(float)
    charging = data.cc * float(np.sum(fly_dists))
    transport = data.ch * float(np.sum(sol.needs_transport))

    return CostBreakdown(
        build=build,
        maintenance=maintenance,
        charging=charging,
        transport=transport,
    )


def robot_distances(sol: Solution, data: ProblemData) -> np.ndarray:
    """Return the (n_robots,) vector of distances to assigned stations."""
    dmat = distance_matrix(
        data.robot_lat, data.robot_lon,
        sol.station_lat, sol.station_lon,
    )
    return dmat[np.arange(data.n_robots), sol.assignments]


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def check_feasibility(sol: Solution, data: ProblemData) -> tuple[bool, list[str]]:
    """Verify all constraints.

    Checks:
        1. Every robot assigned to a valid station.
        2. Station capacity: assigned robots ≤ n_chargers[j] × q.
        3. Charger limit: n_chargers[j] ≤ m.
        4. Transport flags consistent with distances vs ranges.

    Returns:
        (is_feasible, list_of_violation_strings)
    """
    violations = []

    # 1. Assignment validity
    if len(sol.assignments) != data.n_robots:
        violations.append(
            f"Assignment length {len(sol.assignments)} != n_robots {data.n_robots}")

    bad = (sol.assignments < 0) | (sol.assignments >= sol.n_stations)
    if np.any(bad):
        violations.append(f"{int(np.sum(bad))} robots have invalid station index")

    # 2. Capacity constraint
    counts = np.bincount(sol.assignments, minlength=sol.n_stations)
    capacities = sol.n_chargers * data.q
    over = counts > capacities
    for j in np.where(over)[0]:
        violations.append(
            f"Station {j}: {counts[j]} assigned > capacity {capacities[j]} "
            f"({sol.n_chargers[j]}×{data.q})")

    # 3. Charger limit
    over_m = sol.n_chargers > data.m
    for j in np.where(over_m)[0]:
        violations.append(
            f"Station {j}: {sol.n_chargers[j]} chargers > m={data.m}")

    # 4. Transport consistency
    dists = robot_distances(sol, data)
    expected_transport = dists > data.robot_range
    mismatch = sol.needs_transport != expected_transport
    if np.any(mismatch):
        violations.append(
            f"{int(np.sum(mismatch))} robots have inconsistent transport flags")

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# In-place helpers (used by heuristics)
# ---------------------------------------------------------------------------

def update_transport_flags(sol: Solution, data: ProblemData) -> None:
    """Set needs_transport based on distance vs deterministic range. In-place."""
    dists = robot_distances(sol, data)
    sol.needs_transport = dists > data.robot_range


def update_charger_counts(sol: Solution, data: ProblemData) -> None:
    """Set n_chargers to the minimum needed, capped at m. In-place."""
    counts = np.bincount(sol.assignments, minlength=sol.n_stations)
    sol.n_chargers = np.minimum(
        np.ceil(counts / data.q).astype(int), data.m)


def remove_empty_stations(sol: Solution) -> None:
    """Remove stations with zero assigned robots. In-place.

    Re-indexes assignments accordingly.
    """
    counts = np.bincount(sol.assignments, minlength=sol.n_stations)
    keep = counts > 0
    if np.all(keep):
        return

    # Build index mapping: old station index -> new station index
    new_idx = np.full(sol.n_stations, -1, dtype=int)
    new_idx[keep] = np.arange(int(np.sum(keep)))

    sol.station_lon = sol.station_lon[keep]
    sol.station_lat = sol.station_lat[keep]
    sol.n_chargers = sol.n_chargers[keep]
    sol.assignments = new_idx[sol.assignments]
