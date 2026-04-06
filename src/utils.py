"""
Utility functions: Haversine distance and charging probability.

All distance computations use the Haversine (great-circle) formula in miles,
which correctly handles the meridian convergence at Antarctic latitudes
(1° longitude ≈ 19 km at 80°S vs 1° latitude ≈ 111 km).
"""

import numpy as np

EARTH_RADIUS_MILES = 3958.8


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles via the Haversine formula.

    All inputs can be scalars or numpy arrays (broadcasting supported).

    Args:
        lat1, lon1: Latitude/longitude of point(s) 1 in degrees.
        lat2, lon2: Latitude/longitude of point(s) 2 in degrees.

    Returns:
        Distance(s) in miles.
    """
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def distance_matrix(robot_lat, robot_lon, station_lat, station_lon):
    """Compute the (n_robots × n_stations) distance matrix in miles.

    Uses vectorised broadcasting — no explicit loops.

    Args:
        robot_lat:   (n_robots,) array.
        robot_lon:   (n_robots,) array.
        station_lat: (n_stations,) array.
        station_lon: (n_stations,) array.

    Returns:
        (n_robots, n_stations) ndarray of distances in miles.
    """
    return haversine(
        robot_lat[:, np.newaxis], robot_lon[:, np.newaxis],
        station_lat[np.newaxis, :], station_lon[np.newaxis, :],
    )


def charging_probability(ranges, lam=0.012, r_min=10.0):
    """Probability that a robot requires charging.

    p(r) = exp(-λ² (r - r_min)²)

    Ranges close to r_min yield p ≈ 1; large ranges yield p ≈ 0.

    Args:
        ranges: Robot range(s), scalar or array.
        lam:    Decay parameter λ.
        r_min:  Minimum range.

    Returns:
        Charging probability, same shape as `ranges`.
    """
    return np.exp(-(lam ** 2) * (ranges - r_min) ** 2)
