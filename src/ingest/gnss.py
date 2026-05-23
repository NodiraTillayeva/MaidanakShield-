"""
Ingest the *real* GPS constellation from Celestrak two-line elements (TLEs) and
compute, for any place and time, which satellites are actually above the horizon
and how good their geometry is.

This is the physical backbone of the Maidanak Observatory layer: a spoofer can
forge any RF signal, but it cannot move a real satellite. Public ephemeris tells
us the true sky. If a receiver's claimed fix implies a satellite geometry that is
inconsistent with the true constellation (too few real sats, impossible DOP), the
fix is suspect - and the optical observatory can independently confirm where the
satellites really are.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

TLE_CACHE = config.DATA_DIR / "gps_tle.txt"

_TS = None  # lazily-built skyfield timescale (bundled data, no download)


def _timescale():
    global _TS
    if _TS is None:
        from skyfield.api import load
        _TS = load.timescale(builtin=True)
    return _TS


def fetch_gps_tle(use_cache: bool = False) -> str:
    """Return the raw GPS-operational TLE text, caching it to disk."""
    if use_cache and TLE_CACHE.exists():
        return TLE_CACHE.read_text()
    try:
        r = requests.get(config.CELESTRAK_GPS_TLE_URL, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        text = r.text
        if "1 " in text and "2 " in text:  # sanity: looks like TLE
            TLE_CACHE.write_text(text)
            n = text.count("\n1 ")
            print(f"[gnss] fetched live GPS TLEs (~{n} satellites)")
            return text
    except Exception as exc:  # noqa: BLE001
        print(f"[gnss] live TLE fetch failed ({exc}); using cache if present")
    if TLE_CACHE.exists():
        return TLE_CACHE.read_text()
    raise RuntimeError("No GPS TLE data available (no network and no cache).")


def load_satellites(use_cache: bool = False):
    """Parse the TLE text into skyfield EarthSatellite objects."""
    from skyfield.api import EarthSatellite

    text = fetch_gps_tle(use_cache=use_cache)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    ts = _timescale()
    sats = []
    i = 0
    while i + 2 < len(lines) + 1:
        if i + 2 >= len(lines):
            break
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                sats.append(EarthSatellite(l1, l2, name, ts))
            except Exception:  # noqa: BLE001 - skip malformed records
                pass
            i += 3
        else:
            i += 1
    return sats


def visible_satellites(lat: float, lon: float, elevation_m: float,
                       when: datetime | None = None,
                       mask_deg: float = 5.0,
                       use_cache: bool = False) -> pd.DataFrame:
    """Satellites above the elevation mask, with azimuth/elevation and ENU unit vectors."""
    from skyfield.api import wgs84

    when = when or datetime.now(timezone.utc)
    ts = _timescale()
    t = ts.from_datetime(when.astimezone(timezone.utc))
    observer = wgs84.latlon(lat, lon, elevation_m)

    rows = []
    for sat in load_satellites(use_cache=use_cache):
        topo = (sat - observer).at(t)
        alt, az, dist = topo.altaz()
        el_deg = alt.degrees
        if el_deg < mask_deg:
            continue
        el, azr = np.radians(el_deg), np.radians(az.degrees)
        rows.append({
            "name": sat.name,
            "az_deg": az.degrees,
            "el_deg": el_deg,
            "range_km": dist.km,
            # local East-North-Up unit vector toward the satellite
            "e": np.cos(el) * np.sin(azr),
            "n": np.cos(el) * np.cos(azr),
            "u": np.sin(el),
        })
    return pd.DataFrame(rows)


def _dop_from_unit_vectors(e, n, u) -> dict:
    """Dilution-of-precision metrics from satellite ENU unit vectors."""
    if len(e) < 4:
        return {"gdop": np.nan, "pdop": np.nan, "hdop": np.nan, "vdop": np.nan}
    G = np.column_stack([-np.asarray(e), -np.asarray(n), -np.asarray(u), np.ones(len(e))])
    try:
        Q = np.linalg.inv(G.T @ G)
    except np.linalg.LinAlgError:
        return {"gdop": np.nan, "pdop": np.nan, "hdop": np.nan, "vdop": np.nan}
    d = np.diag(Q)
    return {
        "gdop": float(np.sqrt(d.sum())),
        "pdop": float(np.sqrt(d[0] + d[1] + d[2])),
        "hdop": float(np.sqrt(d[0] + d[1])),
        "vdop": float(np.sqrt(d[2])),
    }


def constellation_geometry(lat: float, lon: float, elevation_m: float = 0.0,
                           when: datetime | None = None,
                           use_cache: bool = False) -> dict:
    """Summarise the true GPS sky over a point: count, mean elevation, DOP."""
    vis = visible_satellites(lat, lon, elevation_m, when=when, use_cache=use_cache)
    out = {"n_visible": int(len(vis)),
           "mean_elevation_deg": float(vis["el_deg"].mean()) if len(vis) else np.nan,
           "satellites": vis}
    out.update(_dop_from_unit_vectors(vis.get("e", []), vis.get("n", []), vis.get("u", [])))
    return out


if __name__ == "__main__":
    obs = config.MAIDANAK_OBSERVATORY
    geo = constellation_geometry(obs["lat"], obs["lon"], obs["elevation_m"])
    print(f"Real GPS sky over {obs['name']} now: "
          f"{geo['n_visible']} sats visible, PDOP={geo['pdop']:.2f}, "
          f"mean elevation {geo['mean_elevation_deg']:.1f} deg")
