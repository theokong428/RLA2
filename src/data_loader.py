"""
Data loading and preprocessing for the Antarctic charging station problem.

Reads robot locations, deterministic ranges, scenario ranges, and parameters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProblemData:
    """Container for all problem data and parameters."""
    # Robot positions
    n_robots: int
    robot_lon: np.ndarray   # (n_robots,)
    robot_lat: np.ndarray   # (n_robots,)

    # Ranges
    robot_range: np.ndarray     # (n_robots,) deterministic range (miles)
    n_scenarios: int
    scenario_range: np.ndarray  # (n_robots, n_scenarios) stochastic ranges

    # Cost parameters
    m: int          # max chargers per station
    q: int          # max robots per charger (station capacity = m * q)
    cb: float       # build cost per station (£/station)
    ch: float       # human transport cost (£/robot)
    cm: float       # maintenance cost per charger (£/charger)
    cc: float       # charging cost per mile (£/mile)

    # Charging probability parameters
    lam: float      # lambda decay parameter
    r_min: float    # minimum possible range (miles)
    r_max: float    # maximum range after full charge (miles)


def load_data(data_dir: str = None) -> ProblemData:
    """Load all problem data from CSV files and return a ProblemData object.

    Args:
        data_dir: Path to the data directory. If None, auto-resolves to
                  <project_root>/data regardless of working directory.

    Returns:
        ProblemData with all loaded data.
    """
    if data_dir is None:
        # Auto-resolve: this file lives in src/, data is in ../data/
        data_path = Path(__file__).resolve().parent.parent / "data"
    else:
        data_path = Path(data_dir)

    # --- Robot locations ---
    locations = pd.read_csv(data_path / "robot_locations.csv")
    robot_lon = locations["longitude"].values.astype(np.float64)
    robot_lat = locations["latitude"].values.astype(np.float64)
    n_robots = len(locations)

    # --- Scenario ranges (s0 = deterministic range) ---
    scenarios = pd.read_csv(data_path / "range_scenarios.csv", index_col=0)
    scenario_range = scenarios.values.astype(np.float64)  # (n_robots, 100)
    n_scenarios = scenario_range.shape[1]
    robot_range = scenario_range[:, 0].copy()  # s0 is the deterministic range

    return ProblemData(
        n_robots=n_robots,
        robot_lon=robot_lon,
        robot_lat=robot_lat,
        robot_range=robot_range,
        n_scenarios=n_scenarios,
        scenario_range=scenario_range,
        m=8,
        q=2,
        cb=5000.0,
        ch=1000.0,
        cm=500.0,
        cc=0.42,
        lam=0.012,
        r_min=10.0,
        r_max=175.0,
    )


def get_subset(data: ProblemData, indices: np.ndarray) -> ProblemData:
    """Extract a subset of robots for small-instance testing.

    Args:
        data: Full problem data.
        indices: Array of robot indices to include.

    Returns:
        New ProblemData containing only the specified robots.
    """
    idx = np.asarray(indices)
    return ProblemData(
        n_robots=len(idx),
        robot_lon=data.robot_lon[idx].copy(),
        robot_lat=data.robot_lat[idx].copy(),
        robot_range=data.robot_range[idx].copy(),
        n_scenarios=data.n_scenarios,
        scenario_range=data.scenario_range[idx].copy(),
        m=data.m, q=data.q,
        cb=data.cb, ch=data.ch, cm=data.cm, cc=data.cc,
        lam=data.lam, r_min=data.r_min, r_max=data.r_max,
    )
