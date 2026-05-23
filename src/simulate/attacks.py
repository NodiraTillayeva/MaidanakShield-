"""
Attack injector - known spoofing/jamming signatures planted into baseline tracks.

These four signatures cover what is actually observed in the wild and in the
GNSS-security literature, and each stresses a different detector cue:

  spoof_pulloff : the covert attack - the reported position is slowly walked off
                  the true path, so heading/velocity stop agreeing with motion.
  spoof_teleport: replay/meaconing - the fix jumps to a decoy location.
  spoof_circle  : the signature behind the famous "aircraft flying in circles"
                  spoofing incidents near contested airfields.
  jam_dropout   : jamming - the receiver loses lock and the position freezes while
                  the aircraft keeps moving, so implied speed collapses to zero.

Every injected sample is labelled (is_attack, attack_type) so detection can be
scored objectively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.geo import destination_point, haversine_km

ATTACK_TYPES = ["spoof_pulloff", "spoof_teleport", "spoof_circle", "jam_dropout"]


def _apply_spoof_pulloff(g, onset, rng):
    drift_ms = rng.uniform(7, 55)  # lateral pull-off rate (m/s) - some are very covert
    base_heading = g["heading"].iloc[onset]
    for i in range(onset, len(g)):
        dt = g["t"].iloc[i] - g["t"].iloc[onset]
        offset_km = drift_ms * dt / 1000.0
        lat, lon = destination_point(g["lat"].iat[i], g["lon"].iat[i],
                                     (base_heading + 90) % 360, offset_km)
        g.iat[i, g.columns.get_loc("lat")] = lat
        g.iat[i, g.columns.get_loc("lon")] = lon
    return g


def _apply_spoof_teleport(g, onset, rng):
    jump_km = rng.uniform(60, 180)
    brg = rng.uniform(0, 360)
    for i in range(onset, len(g)):
        lat, lon = destination_point(g["lat"].iat[i], g["lon"].iat[i], brg, jump_km)
        g.iat[i, g.columns.get_loc("lat")] = lat
        g.iat[i, g.columns.get_loc("lon")] = lon
    return g


def _apply_spoof_circle(g, onset, rng):
    radius_km = rng.uniform(1.0, 3.0)
    clat, clon = g["lat"].iat[onset], g["lon"].iat[onset]
    omega = rng.uniform(14, 110)  # deg of circle per sample (some tight, some very gentle)
    for j, i in enumerate(range(onset, len(g))):
        ang = (j * omega) % 360
        lat, lon = destination_point(clat, clon, ang, radius_km)
        g.iat[i, g.columns.get_loc("lat")] = lat
        g.iat[i, g.columns.get_loc("lon")] = lon
        g.iat[i, g.columns.get_loc("heading")] = (ang + 90) % 360
    return g


def _apply_jam_dropout(g, onset, rng):
    # receiver loses lock: position freezes (sometimes only partially) while
    # reported velocity stays high. Partial dropouts are harder to catch.
    flat, flon = g["lat"].iat[onset], g["lon"].iat[onset]
    leak = rng.uniform(0.0, 0.80)   # fraction of true motion that still leaks through
    for i in range(onset, len(g)):
        tl = g["lat"].iat[i] if i < len(g) else flat
        to = g["lon"].iat[i] if i < len(g) else flon
        g.iat[i, g.columns.get_loc("lat")] = flat + leak * (tl - flat) + rng.normal(0, 1e-4)
        g.iat[i, g.columns.get_loc("lon")] = flon + leak * (to - flon) + rng.normal(0, 1e-4)
    return g


_DISPATCH = {
    "spoof_pulloff": _apply_spoof_pulloff,
    "spoof_teleport": _apply_spoof_teleport,
    "spoof_circle": _apply_spoof_circle,
    "jam_dropout": _apply_jam_dropout,
}


def inject_attacks(baseline: pd.DataFrame,
                   fraction: float = 0.3,
                   types: list[str] | None = None,
                   seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Plant attacks into a fraction of the baseline tracks.

    Returns (modified_long_df, truth_table). The truth table has one row per
    aircraft: [track_id, attack_type, is_attack(bool), onset_t].
    """
    types = types or ATTACK_TYPES
    rng = np.random.default_rng(seed)
    df = baseline.copy()
    ids = df["track_id"].unique()
    n_attack = max(1, int(round(len(ids) * fraction)))
    attacked = set(rng.choice(ids, size=n_attack, replace=False))

    truth = {tid: {"track_id": tid, "attack_type": "none",
                   "is_attack": False, "onset_t": np.nan} for tid in ids}

    pieces = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.reset_index(drop=True)
        if tid in attacked:
            atype = rng.choice(types)
            onset = rng.integers(low=max(2, len(g) // 5), high=max(3, len(g) // 2))
            g = _DISPATCH[atype](g, int(onset), rng)
            g.loc[onset:, "is_attack"] = True
            g.loc[onset:, "attack_type"] = atype
            truth[tid].update(attack_type=atype, is_attack=True,
                              onset_t=float(g["t"].iloc[onset]))
        pieces.append(g)

    out = pd.concat(pieces, ignore_index=True)
    truth_df = pd.DataFrame(truth.values())
    return out, truth_df


def apply_jammer(tracks: pd.DataFrame, jlat: float, jlon: float,
                 radius_km: float = 45.0, decoy_spread_km: float = 8.0,
                 seed: int = 23) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model a ground spoofer that drags every aircraft inside `radius_km` onto a
    false decoy position near itself - the mechanism that produces a spatial
    interference *zone* (many aircraft, one place).

    Returns (modified_tracks, truth_table). Truth rows use attack_type 'spoof_decoy'.
    """
    rng = np.random.default_rng(seed)
    df = tracks.copy()
    truth_rows = {}
    pieces = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.reset_index(drop=True)
        true_d = haversine_km(g["lat"].to_numpy(), g["lon"].to_numpy(), jlat, jlon)
        in_zone = np.where(true_d <= radius_km)[0]
        if len(in_zone):
            onset = int(in_zone[0])
            for i in range(onset, len(g)):
                # everyone is reported at ~the same false location -> tight cluster
                off_brg = rng.uniform(0, 360)
                off_km = rng.uniform(0, decoy_spread_km)
                lat, lon = destination_point(jlat, jlon, off_brg, off_km)
                g.iat[i, g.columns.get_loc("lat")] = lat
                g.iat[i, g.columns.get_loc("lon")] = lon
            g.loc[onset:, "is_attack"] = True
            g.loc[onset:, "attack_type"] = "spoof_decoy"
            truth_rows[tid] = {"track_id": tid, "attack_type": "spoof_decoy",
                               "is_attack": True, "onset_t": float(g["t"].iloc[onset])}
        else:
            truth_rows[tid] = {"track_id": tid, "attack_type": "none",
                               "is_attack": False, "onset_t": np.nan}
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    return out, pd.DataFrame(truth_rows.values())


if __name__ == "__main__":
    from src.simulate.scenario import generate_baseline
    base = generate_baseline()
    atk, truth = inject_attacks(base)
    print(truth["attack_type"].value_counts())
