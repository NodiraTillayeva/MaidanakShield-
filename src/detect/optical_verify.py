"""
Maidanak Observatory - independent physical ground truth.

A spoofer can forge any RF signal, but it cannot move a real satellite. The
observatory layer uses the *true* GPS constellation (from public ephemeris, and in
the full system, optical confirmation of satellite positions) to answer one
question that no RF-only competitor can: **is degraded navigation explainable by
the real sky, or is something injecting it?**

If the true constellation over an area is healthy (plenty of well-distributed
satellites, low DOP) yet aircraft there show spoof-like physics, that is
high-confidence interference - there is no natural reason for the fix to be bad.
If the real sky is genuinely poor, the same physics anomaly is more ambiguous.
This is the cue that turns "looks weird" into "verified against ground truth".
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

import config
from src.ingest import gnss


def sky_health(lat: float, lon: float, when: datetime | None = None,
               elevation_m: float = 0.0, use_cache: bool = True) -> dict:
    """Health of the *real* GPS sky over a point.

    Returns the raw constellation metrics plus `corroboration` in [0, 1]:
    1.0 means the sky is excellent, so any navigation anomaly there is highly
    suspicious; lower means poor real geometry could itself explain trouble.
    """
    geo = gnss.constellation_geometry(lat, lon, elevation_m, when=when, use_cache=use_cache)
    n = geo["n_visible"]
    pdop = geo["pdop"]

    # plenty of sats -> healthy; PDOP under ~3 is good, over ~6 is poor.
    sat_factor = float(np.clip((n - 4) / 6.0, 0.0, 1.0))
    if pdop != pdop:  # NaN -> cannot even form a fix, sky is effectively unusable
        dop_factor = 0.0
    else:
        dop_factor = float(np.clip((6.0 - pdop) / 4.0, 0.0, 1.0))
    geo["corroboration"] = round(0.5 * sat_factor + 0.5 * dop_factor, 3)
    return geo


def confirm_zone(lat: float, lon: float, when: datetime | None = None,
                 use_cache: bool = True) -> dict:
    """Forensic confirmation for a flagged interference zone.

    Produces a human-readable verdict comparing the real sky against the
    observed interference - the kind of statement that goes into an incident report.
    """
    when = when or datetime.now(timezone.utc)
    h = sky_health(lat, lon, when=when, use_cache=use_cache)
    if h["corroboration"] >= 0.6:
        verdict = ("Real GPS sky is HEALTHY here "
                   f"({h['n_visible']} sats, PDOP {h['pdop']:.1f}). "
                   "Navigation anomalies are NOT explained by geometry -> "
                   "consistent with deliberate interference.")
        confidence = "high"
    elif h["corroboration"] >= 0.3:
        verdict = ("Real GPS sky is MARGINAL here "
                   f"({h['n_visible']} sats, PDOP {h['pdop']:.1f}). "
                   "Interference likely but geometry is a partial factor.")
        confidence = "medium"
    else:
        verdict = ("Real GPS sky is POOR here "
                   f"({h['n_visible']} sats). Degraded navigation may be partly natural.")
        confidence = "low"
    return {"lat": lat, "lon": lon, "when": when.isoformat(),
            "n_visible": h["n_visible"], "pdop": h["pdop"],
            "corroboration": h["corroboration"],
            "confidence": confidence, "verdict": verdict}


def apply_optical_cue(track_summary, region: config.Region = config.DEFAULT_REGION,
                      when: datetime | None = None, use_cache: bool = True):
    """Fuse the regional optical corroboration into per-aircraft scores.

    We compute one real-sky snapshot at the region centre (the constellation barely
    changes across the box at a single instant) and use it to either reinforce or
    temper each aircraft's physics score, then recompute the flag.
    """
    if track_summary.empty:
        return track_summary, None
    clat, clon = region.center
    h = sky_health(clat, clon, when=when, use_cache=use_cache)
    corr = h["corroboration"]
    w = config.DETECT.weight_optical
    out = track_summary.copy()
    # optical cue scales the physics evidence: healthy sky amplifies suspicion.
    out["optical_corroboration"] = corr
    out["score"] = (1 - w) * out["peak_score"] + w * (out["peak_score"] * corr)
    out["flagged"] = out["score"] >= config.DETECT.alert_threshold
    return out.sort_values("score", ascending=False).reset_index(drop=True), h


if __name__ == "__main__":
    obs = config.MAIDANAK_OBSERVATORY
    print(confirm_zone(obs["lat"], obs["lon"])["verdict"])
