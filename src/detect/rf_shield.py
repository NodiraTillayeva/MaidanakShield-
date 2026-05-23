"""
Maidanak Shield - the ground/airspace-level interference map.

Individual aircraft anomalies are noisy; a jamming or spoofing source reveals
itself when *many* aircraft go bad in the *same place*. The Shield rasterises the
airspace into a grid and lights up a cell as an interference zone when enough
independent aircraft report spoof-like behaviour inside it. This is the same
principle GPSJAM uses on global ADS-B integrity data, reduced to a corridor and
fused with the Sentinel Twin's physics evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

DET = config.DETECT


def interference_zones(scored: pd.DataFrame,
                       grid_deg: float | None = None,
                       min_flags: int | None = None,
                       threshold: float | None = None) -> pd.DataFrame:
    """Aggregate flagged samples into interference zones.

    Returns a DataFrame of zones: cell centre lat/lon, number of distinct aircraft
    flagged in the cell, total flagged samples, mean spoofing score, and a 0..1
    severity. Empty if nothing crosses the detection bar.
    """
    grid_deg = grid_deg or DET.zone_grid_deg
    min_flags = min_flags or DET.zone_min_flags
    threshold = threshold if threshold is not None else DET.alert_threshold
    if scored.empty:
        return pd.DataFrame(columns=["lat", "lon", "n_aircraft", "n_samples",
                                     "mean_score", "severity"])

    hot = scored[scored["physics_score"] >= threshold].copy()
    if hot.empty:
        return pd.DataFrame(columns=["lat", "lon", "n_aircraft", "n_samples",
                                     "mean_score", "severity"])

    hot["cell_lat"] = (np.floor(hot["lat"] / grid_deg) * grid_deg + grid_deg / 2).round(4)
    hot["cell_lon"] = (np.floor(hot["lon"] / grid_deg) * grid_deg + grid_deg / 2).round(4)

    grp = hot.groupby(["cell_lat", "cell_lon"])
    zones = grp.agg(
        n_aircraft=("track_id", "nunique"),
        n_samples=("physics_score", "size"),
        mean_score=("physics_score", "mean"),
    ).reset_index().rename(columns={"cell_lat": "lat", "cell_lon": "lon"})

    zones = zones[zones["n_aircraft"] >= min_flags].copy()
    if zones.empty:
        return zones
    # severity blends how many aircraft are affected with how bad the fixes are
    aircraft_term = np.clip(zones["n_aircraft"] / 5.0, 0, 1)
    zones["severity"] = (0.6 * aircraft_term + 0.4 * zones["mean_score"]).round(3)
    return zones.sort_values("severity", ascending=False).reset_index(drop=True)


def zones_from_points(df: pd.DataFrame,
                      flag_col: str = "flagged",
                      score_col: str = "integrity_score",
                      grid_deg: float | None = None,
                      min_count: int | None = None) -> pd.DataFrame:
    """Cluster *real* flagged aircraft (degraded GNSS integrity) into zones.

    Generalises the Shield to live data: a cell becomes an interference zone when
    enough degraded aircraft fall inside it. Severity blends how many aircraft are
    hit with how degraded they are and the local degraded *fraction*.
    """
    grid_deg = grid_deg or DET.zone_grid_deg
    min_count = min_count or DET.zone_min_flags
    cols = ["lat", "lon", "n_aircraft", "n_samples", "mean_score", "severity"]
    if df.empty or flag_col not in df:
        return pd.DataFrame(columns=cols)

    d = df.copy()
    d["cell_lat"] = (np.floor(d["lat"] / grid_deg) * grid_deg + grid_deg / 2).round(4)
    d["cell_lon"] = (np.floor(d["lon"] / grid_deg) * grid_deg + grid_deg / 2).round(4)
    flagged = d[d[flag_col]]
    if flagged.empty:
        return pd.DataFrame(columns=cols)

    # total aircraft per cell (for the degraded fraction)
    total = d.groupby(["cell_lat", "cell_lon"]).size().rename("total")
    grp = flagged.groupby(["cell_lat", "cell_lon"])
    zones = grp.agg(n_aircraft=("track_id", "nunique"),
                    n_samples=("track_id", "size"),
                    mean_score=(score_col, "mean")).reset_index()
    zones = zones.merge(total.reset_index(), on=["cell_lat", "cell_lon"])
    zones = zones.rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    zones = zones[zones["n_aircraft"] >= min_count].copy()
    if zones.empty:
        return pd.DataFrame(columns=cols)
    frac = zones["n_aircraft"] / zones["total"]
    count_term = np.clip(zones["n_aircraft"] / 8.0, 0, 1)
    zones["severity"] = (0.5 * frac + 0.3 * count_term
                         + 0.2 * zones["mean_score"].fillna(0.5)).round(3)
    return zones[cols].sort_values("severity", ascending=False).reset_index(drop=True)


def zone_label(severity: float) -> str:
    if severity >= 0.66:
        return "SEVERE"
    if severity >= 0.4:
        return "ELEVATED"
    return "WATCH"
