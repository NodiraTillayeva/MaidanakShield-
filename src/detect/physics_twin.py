"""
Sentinel Twin - the onboard digital-twin physics check.

Every other product on the market reacts *after* a spoofed signal has corrupted the
receiver. The Sentinel Twin instead keeps a lightweight kinematic model of the
aircraft - it *dead-reckons* the next position from the last trusted state - and
asks of every new GNSS-derived fix: *could a real aircraft have got here?*

Cues, each normalised to 0..1:

  dr_residual : distance between the reported fix and the twin's own dead-reckoned
                prediction. This single cue catches every signature we care about -
                a slow pull-off, a teleport, a frozen (jammed) fix, or a circling
                spoof all make the report diverge from honest dead reckoning.
  speed       : implied ground speed (from position delta) disagreeing with the
                reported velocity - the classic freeze/teleport tell.
  bearing     : implied direction of travel disagreeing with reported heading.
  teleport    : a position jump larger than is reachable in the elapsed time.
  turn/accel  : manoeuvre rates beyond the airframe envelope.
  altitude    : altitude outside a sane band or changing impossibly fast.

The fused score needs no ground station and no cloud - it runs on the aircraft,
which is why it can warn the crew *before* the corrupted fix is trusted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.geo import haversine_km, bearing_deg, destination_point

PHYS = config.PHYSICS
DET = config.DETECT


def _norm_over(value, limit):
    """0 at/under `limit`, ramping to 1 as the value exceeds it."""
    if limit <= 0:
        return 0.0
    return float(np.clip((np.abs(value) - limit) / limit, 0.0, 1.0))


def _ang_diff(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


def score_track(track: pd.DataFrame) -> pd.DataFrame:
    """Score one aircraft's trajectory; returns it with per-sample cue + score columns."""
    t = track.reset_index(drop=True).copy()
    n = len(t)
    cues = {k: np.zeros(n) for k in
            ["dr_anom", "speed_anom", "bearing_anom", "teleport_anom",
             "turn_anom", "accel_anom", "alt_anom"]}
    implied_speed = np.zeros(n)
    residual_km = np.zeros(n)
    dr_lat_arr = np.zeros(n)
    dr_lon_arr = np.zeros(n)

    # the twin's dead-reckoned position, seeded on the first trusted fix
    dr_lat, dr_lon = t["lat"].iloc[0], t["lon"].iloc[0]
    dr_lat_arr[0], dr_lon_arr[0] = dr_lat, dr_lon

    for i in range(1, n):
        dt = max(t["t"].iloc[i] - t["t"].iloc[i - 1], 1e-3)
        v_prev = t["velocity"].iloc[i - 1]
        h_prev = t["heading"].iloc[i - 1]

        # advance the twin from the previous trusted state (dead reckoning)
        dr_lat, dr_lon = destination_point(dr_lat, dr_lon, h_prev, v_prev * dt / 1000.0)
        dr_lat_arr[i], dr_lon_arr[i] = dr_lat, dr_lon
        res = haversine_km(t["lat"].iloc[i], t["lon"].iloc[i], dr_lat, dr_lon)
        residual_km[i] = res
        cues["dr_anom"][i] = float(np.clip(res / DET.dr_tolerance_km, 0.0, 1.0))

        # implied motion from the reported positions
        dist_km = haversine_km(t["lat"].iloc[i - 1], t["lon"].iloc[i - 1],
                               t["lat"].iloc[i], t["lon"].iloc[i])
        v_implied = dist_km * 1000.0 / dt
        implied_speed[i] = v_implied

        # speed mismatch (implied vs reported), and over-envelope speed
        v_rep = t["velocity"].iloc[i]
        mismatch = abs(v_implied - v_rep) / max(v_rep, 50.0) if v_rep == v_rep else 0.0
        cues["speed_anom"][i] = float(np.clip(
            max(mismatch, _norm_over(v_implied, PHYS.max_speed_ms)), 0.0, 1.0))

        # bearing mismatch (only meaningful if the aircraft actually moved)
        if dist_km > 0.05:
            implied_brg = bearing_deg(t["lat"].iloc[i - 1], t["lon"].iloc[i - 1],
                                      t["lat"].iloc[i], t["lon"].iloc[i])
            cues["bearing_anom"][i] = float(np.clip(
                _ang_diff(implied_brg, t["heading"].iloc[i]) / 90.0, 0.0, 1.0))

        # teleport: jump beyond what the envelope allows in this interval
        reach_km = PHYS.max_speed_ms * dt / 1000.0
        cues["teleport_anom"][i] = _norm_over(max(dist_km - reach_km, 0.0),
                                              max(PHYS.max_teleport_km, reach_km))

        # turn rate / acceleration / altitude
        cues["turn_anom"][i] = _norm_over(
            _ang_diff(t["heading"].iloc[i], h_prev) / dt, PHYS.max_turn_rate_deg_s)
        cues["accel_anom"][i] = _norm_over(
            (implied_speed[i] - implied_speed[i - 1]) / dt, PHYS.max_accel_ms2)
        alt = t["alt"].iloc[i]
        d_alt = abs(alt - t["alt"].iloc[i - 1]) / dt
        out_of_band = alt < PHYS.min_sane_altitude_m or alt > PHYS.max_sane_altitude_m
        cues["alt_anom"][i] = max(_norm_over(d_alt, PHYS.max_climb_ms),
                                  1.0 if out_of_band else 0.0)

    for k, v in cues.items():
        t[k] = v
    t["implied_speed"] = implied_speed
    t["dr_residual_km"] = residual_km
    t["dr_lat"] = dr_lat_arr
    t["dr_lon"] = dr_lon_arr

    t["physics_score"] = (
        DET.weight_dr * t["dr_anom"]
        + DET.weight_speed * t["speed_anom"]
        + DET.weight_bearing * t["bearing_anom"]
        + DET.weight_teleport * t["teleport_anom"]
        + DET.weight_turn * t["turn_anom"]
        + DET.weight_accel * t["accel_anom"]
        + DET.weight_altitude * t["alt_anom"]
    ) / (DET.weight_dr + DET.weight_speed + DET.weight_bearing + DET.weight_teleport
         + DET.weight_turn + DET.weight_accel + DET.weight_altitude)
    return t


def score_all(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return tracks
    out = [score_track(g) for _, g in tracks.groupby("track_id", sort=False)]
    return pd.concat(out, ignore_index=True)


def track_summary(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per aircraft: peak/sustained spoofing score and the flag.

    We flag on a *sustained* score (rolling 2-sample mean) so a single noisy
    sample cannot raise a false alarm, while a real attack - which persists - does.
    """
    if scored.empty:
        return pd.DataFrame()

    rows = []
    for tid, g in scored.groupby("track_id", sort=False):
        g = g.sort_values("t")
        sustained = g["physics_score"].rolling(2, min_periods=1).mean()
        rows.append({
            "track_id": tid,
            "callsign": g["callsign"].iloc[0] if "callsign" in g else tid,
            "lat": g["lat"].iloc[-1], "lon": g["lon"].iloc[-1],
            "peak_score": float(sustained.max()),
            "mean_score": float(g["physics_score"].mean()),
            "n_samples": int(len(g)),
        })
    agg = pd.DataFrame(rows)
    agg["flagged"] = agg["peak_score"] >= DET.alert_threshold
    return agg.sort_values("peak_score", ascending=False).reset_index(drop=True)
