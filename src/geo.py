"""Geodesy helpers shared across the pipeline (great-circle math on a spherical Earth)."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or numpy arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 in degrees [0, 360)."""
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def destination_point(lat, lon, bearing, distance_km):
    """Point reached travelling `distance_km` from (lat, lon) on `bearing` degrees."""
    ang = distance_km / EARTH_RADIUS_KM
    brg = np.radians(bearing)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2 = np.arcsin(np.sin(lat1) * np.cos(ang) + np.cos(lat1) * np.sin(ang) * np.cos(brg))
    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(ang) * np.cos(lat1),
        np.cos(ang) - np.sin(lat1) * np.sin(lat2),
    )
    return np.degrees(lat2), (np.degrees(lon2) + 540.0) % 360.0 - 180.0


def great_circle_points(lat1, lon1, lat2, lon2, n):
    """`n` evenly spaced points along the great circle from start to end (inclusive)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    d = 2 * np.arcsin(
        np.sqrt(
            np.sin((lat2r - lat1r) / 2) ** 2
            + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2
        )
    )
    if d == 0:
        return np.full(n, lat1), np.full(n, lon1)
    f = np.linspace(0, 1, n)
    a = np.sin((1 - f) * d) / np.sin(d)
    b = np.sin(f * d) / np.sin(d)
    x = a * np.cos(lat1r) * np.cos(lon1r) + b * np.cos(lat2r) * np.cos(lon2r)
    y = a * np.cos(lat1r) * np.sin(lon1r) + b * np.cos(lat2r) * np.sin(lon2r)
    z = a * np.sin(lat1r) + b * np.sin(lat2r)
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    lon = np.degrees(np.arctan2(y, x))
    return lat, lon
