"""
Ingest real live air traffic from the OpenSky Network REST API.

OpenSky's anonymous `states/all` endpoint returns the most recent state vector for
every aircraft inside a bounding box. We use it for two things:

  1. A *live snapshot* of the corridor - proof the system runs on real, current data.
  2. *Short real tracks*, assembled by polling a few snapshots a few seconds apart,
     so the Sentinel Twin physics check has genuine trajectories to reason about.

Everything is cached to disk so the dashboard and the demo work offline and are
reproducible. If the network is unavailable we fail soft and let callers fall back
to the synthetic test-bench.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

import config

# Column order returned by the OpenSky states API (see their docs).
_STATE_COLS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]

SNAPSHOT_CACHE = config.DATA_DIR / "opensky_snapshot.csv"
TRACKS_CACHE = config.DATA_DIR / "opensky_tracks.csv"


def fetch_snapshot(region: config.Region = config.DEFAULT_REGION,
                   use_cache: bool = False) -> pd.DataFrame:
    """Return one live snapshot of aircraft inside `region` as a DataFrame.

    Set `use_cache=True` to read the last saved snapshot instead of hitting the API.
    """
    if use_cache and SNAPSHOT_CACHE.exists():
        return pd.read_csv(SNAPSHOT_CACHE)

    params = {"lamin": region.lamin, "lomin": region.lomin,
              "lamax": region.lamax, "lomax": region.lomax}
    try:
        r = requests.get(config.OPENSKY_STATES_URL, params=params,
                         timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except Exception as exc:  # noqa: BLE001 - fail soft on any network/parse error
        print(f"[opensky] live fetch failed ({exc}); using cache if present")
        if SNAPSHOT_CACHE.exists():
            return pd.read_csv(SNAPSHOT_CACHE)
        return pd.DataFrame(columns=_STATE_COLS)

    states = payload.get("states") or []
    df = pd.DataFrame(states, columns=_STATE_COLS)
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    df["callsign"] = df["callsign"].fillna("").str.strip()
    df["snapshot_time"] = payload.get("time")
    df.to_csv(SNAPSHOT_CACHE, index=False)
    print(f"[opensky] live snapshot: {len(df)} aircraft over {region.name}")
    return df


def fetch_tracks(region: config.Region = config.DEFAULT_REGION,
                 n_polls: int = 6,
                 interval_s: float = 12.0,
                 use_cache: bool = False) -> pd.DataFrame:
    """Assemble short real tracks by polling several snapshots.

    Returns a long DataFrame: one row per (aircraft, sample) with columns
    [track_id, t, lat, lon, alt, velocity, heading, vertical_rate]. `track_id`
    is the ICAO24 address so samples of the same airframe line up over time.
    """
    if use_cache and TRACKS_CACHE.exists():
        return pd.read_csv(TRACKS_CACHE)

    rows: list[dict] = []
    t0 = None
    for i in range(n_polls):
        snap = fetch_snapshot(region, use_cache=False)
        if snap.empty:
            break
        stamp = snap["snapshot_time"].iloc[0] if "snapshot_time" in snap else time.time()
        t0 = t0 if t0 is not None else stamp
        for _, s in snap.iterrows():
            if pd.isna(s["velocity"]):
                continue
            rows.append({
                "track_id": s["icao24"],
                "callsign": s["callsign"],
                "t": float(stamp - t0),
                "lat": float(s["latitude"]),
                "lon": float(s["longitude"]),
                "alt": float(s["geo_altitude"]) if pd.notna(s["geo_altitude"])
                else (float(s["baro_altitude"]) if pd.notna(s["baro_altitude"]) else 10000.0),
                "velocity": float(s["velocity"]),
                "heading": float(s["true_track"]) if pd.notna(s["true_track"]) else 0.0,
                "vertical_rate": float(s["vertical_rate"]) if pd.notna(s["vertical_rate"]) else 0.0,
            })
        if i < n_polls - 1:
            time.sleep(interval_s)

    df = pd.DataFrame(rows)
    if not df.empty:
        # keep only airframes seen in at least 3 snapshots → usable trajectories
        counts = df.groupby("track_id")["t"].count()
        keep = counts[counts >= 3].index
        df = df[df["track_id"].isin(keep)].sort_values(["track_id", "t"]).reset_index(drop=True)
        df.to_csv(TRACKS_CACHE, index=False)
        print(f"[opensky] assembled {df['track_id'].nunique()} real tracks "
              f"({len(df)} samples) over {region.name}")
    return df


if __name__ == "__main__":
    snap = fetch_snapshot()
    print(snap[["icao24", "callsign", "latitude", "longitude", "geo_altitude"]].head())
