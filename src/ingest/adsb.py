"""
Real ADS-B ingest *with navigation-integrity* (NIC / NACp / Rc).

OpenSky's basic feed gives positions but not integrity. The community feeds
adsb.lol and airplanes.live expose the raw ADS-B integrity fields - NIC
(Navigation Integrity Category), NACp (Navigation Accuracy Category - position)
and Rc (radius of containment). These are the *real* fingerprint of GNSS
jamming/spoofing: an aircraft flying through interference keeps transmitting but
its receiver reports collapsed integrity (NIC→0). This is the same signal GPSJAM
uses, and it lets Maidanak Sentinel detect *real* interference with no simulation.

Both feeds are free and unauthenticated. We try adsb.lol, fall back to
airplanes.live, and cache every pull so the demo is reproducible offline.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

import config

KT_TO_MS = 0.514444
FT_TO_M = 0.3048

FEEDS = {
    "adsb.lol": "https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{dist}",
    "airplanes.live": "https://api.airplanes.live/v2/point/{lat}/{lon}/{dist}",
}


def _norm(a: dict) -> dict | None:
    """Normalise one raw aircraft record to our schema. None if no usable position."""
    if "lat" not in a or "lon" not in a:
        return None
    nic = a.get("nic")
    nacp = a.get("nac_p")
    gs = a.get("gs")
    alt = a.get("alt_geom", a.get("alt_baro"))
    return {
        "track_id": a.get("hex", ""),
        "callsign": (a.get("flight") or "").strip() or a.get("hex", ""),
        "lat": float(a["lat"]),
        "lon": float(a["lon"]),
        "alt_m": float(alt) * FT_TO_M if isinstance(alt, (int, float)) else float("nan"),
        "velocity": float(gs) * KT_TO_MS if isinstance(gs, (int, float)) else float("nan"),
        "heading": float(a.get("track")) if isinstance(a.get("track"), (int, float)) else 0.0,
        "nic": int(nic) if isinstance(nic, int) else None,
        "nac_p": int(nacp) if isinstance(nacp, int) else None,
        "rc_m": float(a.get("rc")) if isinstance(a.get("rc"), (int, float)) else None,
        "seen_pos": float(a.get("seen_pos")) if isinstance(a.get("seen_pos"), (int, float)) else None,
    }


def fetch_region(lat: float, lon: float, dist_nm: float = 250.0,
                 use_cache: bool = False, cache_tag: str = "live",
                 feed: str | None = None) -> pd.DataFrame:
    """Pull every aircraft (with integrity) within `dist_nm` of a point.

    `feed` forces a specific source ("adsb.lol" / "airplanes.live"); otherwise we
    try each in turn. Forcing a feed is how we get an independent cross-check.
    """
    cache = config.DATA_DIR / f"adsb_{cache_tag}.csv"
    if use_cache and cache.exists():
        return pd.read_csv(cache)

    feeds = {feed: FEEDS[feed]} if feed in FEEDS else FEEDS
    for feed, url in feeds.items():
        try:
            r = requests.get(url.format(lat=lat, lon=lon, dist=int(dist_nm)),
                             timeout=config.HTTP_TIMEOUT + 10,
                             headers={"User-Agent": "MaidanakSentinel/1.0"})
            r.raise_for_status()
            raw = r.json().get("ac") or []
            rows = [x for x in (_norm(a) for a in raw) if x]
            if rows:
                df = pd.DataFrame(rows)
                df["feed"] = feed
                df.to_csv(cache, index=False)
                return df
        except Exception as exc:  # noqa: BLE001 - try next feed
            print(f"[adsb] {feed} failed ({exc})")
    if cache.exists():
        return pd.read_csv(cache)
    return pd.DataFrame()


def scan_watch_regions(use_cache: bool = False) -> pd.DataFrame:
    """Poll every watch region and summarise live interference for each."""
    from src.detect.integrity import classify  # local import to avoid cycle
    rows = []
    for key, wr in config.WATCH_REGIONS.items():
        df = fetch_region(wr.lat, wr.lon, wr.dist_nm,
                          use_cache=use_cache, cache_tag=key)
        if df.empty:
            rows.append({"region": wr.name, "key": key, "aircraft": 0,
                         "degraded": 0, "pct_degraded": 0.0})
            continue
        c = classify(df)
        air = c[c["airborne"]]
        deg = int(air["flagged"].sum())
        rows.append({"region": wr.name, "key": key, "aircraft": int(len(air)),
                     "degraded": deg,
                     "pct_degraded": round(100 * deg / max(len(air), 1), 1)})
        if not use_cache:
            time.sleep(0.3)  # be polite to the free API
    return pd.DataFrame(rows).sort_values("pct_degraded", ascending=False).reset_index(drop=True)


def fetch_tracks(lat: float, lon: float, dist_nm: float = 250.0,
                 n_polls: int = 5, interval_s: float = 15.0,
                 cache_tag: str = "live") -> pd.DataFrame:
    """Assemble short *real* tracks by polling the feed, for the physics twin."""
    cache = config.DATA_DIR / f"adsb_tracks_{cache_tag}.csv"
    frames = []
    t0 = None
    for i in range(n_polls):
        snap = fetch_region(lat, lon, dist_nm, use_cache=False, cache_tag=f"{cache_tag}_poll")
        if snap.empty:
            break
        now = time.time()
        t0 = t0 or now
        snap = snap.copy()
        snap["t"] = now - t0
        frames.append(snap)
        if i < n_polls - 1:
            time.sleep(interval_s)
    if not frames:
        return pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    counts = df.groupby("track_id")["t"].count()
    df = df[df["track_id"].isin(counts[counts >= 3].index)]
    df = df.sort_values(["track_id", "t"]).reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    print(scan_watch_regions().to_string(index=False))
