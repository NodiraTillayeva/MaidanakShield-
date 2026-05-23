"""
Pre-emptive ground-to-air alerting - the Maidanak Shield -> Sentinel Twin handoff.

This is the capability no competitor has: every other system reacts *after* a
spoofed signal corrupts the receiver. Here, once the Shield declares an
interference zone, we project every aircraft's track forward and warn the ones
that are about to fly into it - so the onboard Sentinel Twin can raise its guard
*before* the corrupted fix ever reaches the sensor fusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.geo import haversine_km, destination_point

ALERT = config.ALERT


def _zone_radius_km(grid_deg: float) -> float:
    """Approximate radius of a raster cell, in km (half-diagonal at mid-latitudes)."""
    return grid_deg * 111.0 / 2 * np.sqrt(2)


def generate_alerts(aircraft: pd.DataFrame,
                    zones: pd.DataFrame,
                    lookahead_min: float = ALERT.lookahead_min,
                    buffer_km: float = ALERT.geofence_buffer_km,
                    step_s: float = 30.0) -> pd.DataFrame:
    """Pre-warn aircraft whose projected path enters any interference zone.

    `aircraft` needs columns [track_id, callsign, lat, lon, heading, velocity].
    Returns one row per (aircraft, zone) alert with the estimated time-to-zone.
    """
    cols = ["callsign", "track_id", "zone_lat", "zone_lon", "zone_severity",
            "eta_min", "distance_km", "action"]
    if aircraft.empty or zones.empty:
        return pd.DataFrame(columns=cols)

    zr = _zone_radius_km(config.DETECT.zone_grid_deg) + buffer_km
    n_steps = int(lookahead_min * 60 / step_s)
    alerts = []

    for _, ac in aircraft.iterrows():
        v = ac.get("velocity", 230.0) or 230.0
        for _, z in zones.iterrows():
            best = None
            for s in range(n_steps + 1):
                t_min = s * step_s / 60.0
                plat, plon = destination_point(ac["lat"], ac["lon"], ac["heading"],
                                               v * (s * step_s) / 1000.0)
                d = haversine_km(plat, plon, z["lat"], z["lon"])
                if d <= zr and (best is None or t_min < best[0]):
                    best = (t_min, d)
                    break  # first entry time is what matters
            if best is not None:
                eta, dist = best
                action = ("HOLD / re-route - entering interference zone"
                          if eta <= 4 else "Arm Sentinel Twin, monitor GNSS integrity")
                alerts.append({
                    "callsign": ac.get("callsign", ac["track_id"]),
                    "track_id": ac["track_id"],
                    "zone_lat": round(float(z["lat"]), 3),
                    "zone_lon": round(float(z["lon"]), 3),
                    "zone_severity": float(z["severity"]),
                    "eta_min": round(float(eta), 1),
                    "distance_km": round(float(dist), 1),
                    "action": action,
                })

    df = pd.DataFrame(alerts, columns=cols)
    return df.sort_values(["eta_min", "zone_severity"], ascending=[True, False]).reset_index(drop=True)


def current_states_from_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """Reduce long-format tracks to each aircraft's latest state for alerting."""
    if tracks.empty:
        return pd.DataFrame(columns=["track_id", "callsign", "lat", "lon", "heading", "velocity"])
    last = tracks.sort_values("t").groupby("track_id").tail(1)
    keep = ["track_id", "lat", "lon", "heading", "velocity"]
    if "callsign" in last:
        keep.insert(1, "callsign")
    return last[keep].reset_index(drop=True)
