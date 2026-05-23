"""
GNSS-integrity classification from real ADS-B (Maidanak Shield, real-data mode).

Turns the raw NIC / NACp / Rc fields broadcast by each aircraft into an
interference verdict. NIC (Navigation Integrity Category) is the key signal: at
cruise a healthy receiver reports NIC 7-8 (containment < 0.1 NM); inside GNSS
jamming the receiver cannot guarantee containment and NIC collapses to 0-1 while
the aircraft keeps transmitting. NACp behaves the same way for accuracy.

Thresholds follow common ADS-B integrity practice (and match what GPSJAM treats
as "low accuracy"). We classify only *airborne* aircraft for interference
statistics, since integrity naturally varies on the ground / in the circuit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# airborne if clearly moving and above circuit altitude
AIRBORNE_MIN_ALT_M = 1500.0
AIRBORNE_MIN_SPEED_MS = 60.0

# integrity bands
NIC_LOST, NIC_DEGRADED = 1, 5      # <=1 lost, <=5 degraded
NACP_LOST, NACP_DEGRADED = 4, 7


def _score_row(nic, nacp):
    """Continuous degradation 0 (healthy) .. 1 (fully lost) from NIC and NACp."""
    parts = []
    if nic is not None and not (isinstance(nic, float) and np.isnan(nic)):
        parts.append(np.clip((7 - nic) / 7.0, 0, 1))
    if nacp is not None and not (isinstance(nacp, float) and np.isnan(nacp)):
        parts.append(np.clip((9 - nacp) / 9.0, 0, 1))
    return float(max(parts)) if parts else np.nan


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Add integrity columns: airborne, integrity_score, status, flagged."""
    if df.empty:
        return df.assign(airborne=[], integrity_score=[], status=[], flagged=[])
    out = df.copy()
    alt = pd.to_numeric(out.get("alt_m"), errors="coerce")
    spd = pd.to_numeric(out.get("velocity"), errors="coerce")
    out["airborne"] = (alt.fillna(0) >= AIRBORNE_MIN_ALT_M) | (spd.fillna(0) >= AIRBORNE_MIN_SPEED_MS)

    nic = pd.to_numeric(out.get("nic"), errors="coerce")
    nacp = pd.to_numeric(out.get("nac_p"), errors="coerce")
    out["integrity_score"] = [
        _score_row(n if not np.isnan(n) else None, p if not np.isnan(p) else None)
        for n, p in zip(nic, nacp)]

    def _status(n, p):
        if (not np.isnan(n) and n <= NIC_LOST) or (not np.isnan(p) and p <= NACP_LOST):
            return "lost"
        if (not np.isnan(n) and n <= NIC_DEGRADED) or (not np.isnan(p) and p <= NACP_DEGRADED):
            return "degraded"
        if np.isnan(n) and np.isnan(p):
            return "unknown"
        return "healthy"

    out["status"] = [_status(n, p) for n, p in zip(nic, nacp)]
    # interference flag: strong, conservative - matches the live-scan definition
    out["flagged"] = ((nic.fillna(99) <= NIC_LOST) | (nacp.fillna(99) <= NACP_LOST))
    return out


def summary(classified: pd.DataFrame) -> dict:
    """Headline real-data statistics over airborne aircraft."""
    if classified.empty:
        return {"aircraft": 0, "degraded": 0, "lost": 0, "pct_degraded": 0.0}
    air = classified[classified["airborne"]]
    deg = air[air["status"].isin(["degraded", "lost"])]
    lost = air[air["status"] == "lost"]
    return {
        "aircraft": int(len(air)),
        "degraded": int(len(deg)),
        "lost": int(len(lost)),
        "pct_degraded": round(100 * len(deg) / max(len(air), 1), 1),
    }
