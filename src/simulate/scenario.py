"""
Synthetic test-bench: realistic baseline traffic for the Central Asian corridor.

Real spoofing events are (thankfully) rare in any given five-minute window of live
ADS-B, which makes them impossible to *measure* a detector against. So we build a
controlled, labelled test-bench: physically plausible cruise tracks seeded inside
the corridor, into which `attacks.py` injects known spoofing/jamming so we can
score the detector with real precision/recall numbers.

The baseline kinematics (cruise speeds, altitudes, GPS position noise) are chosen
to match what the live OpenSky feed actually shows over the region.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.geo import destination_point, bearing_deg

CRUISE_SPEED_MS = (215.0, 250.0)      # typical airliner ground speed band
CRUISE_ALT_M = (9000.0, 11800.0)
GPS_POS_NOISE_M = 15.0                 # honest civilian GPS scatter
HEADING_NOISE_DEG = 0.4


def generate_baseline(n_aircraft: int = 40,
                      samples: int = 30,
                      dt: float = config.DETECT.sample_dt_s,
                      region: config.Region = config.DEFAULT_REGION,
                      seed: int = 7) -> pd.DataFrame:
    """Generate clean cruise tracks inside `region`.

    Long-format DataFrame: [track_id, callsign, t, lat, lon, alt, velocity,
    heading, vertical_rate, is_attack, attack_type]. All baseline samples are
    labelled is_attack=False.
    """
    rng = np.random.default_rng(seed)
    rows = []
    lat_lo, lat_hi = region.lamin + 1.0, region.lamax - 1.0
    lon_lo, lon_hi = region.lomin + 2.0, region.lomax - 2.0

    for k in range(n_aircraft):
        v = rng.uniform(*CRUISE_SPEED_MS)
        alt = rng.uniform(*CRUISE_ALT_M)
        heading = rng.uniform(0, 360)
        lat = rng.uniform(lat_lo, lat_hi)
        lon = rng.uniform(lon_lo, lon_hi)
        callsign = f"SIM{k:03d}"
        track_id = f"sim{k:04d}"

        # ~40% of honest aircraft are manoeuvring (turns / climbs) - these are the
        # realistic edge cases that make detection genuinely non-trivial.
        maneuver = rng.random() < 0.40
        turn_start = int(rng.integers(3, samples - 6)) if maneuver else samples + 1
        turn_len = int(rng.integers(4, 10))
        turn_rate = rng.choice([-1, 1]) * rng.uniform(15, 32)   # deg per sample (real turns)
        vrate = rng.uniform(-12, 12) if (maneuver and rng.random() < 0.5) else 0.0

        for i in range(samples):
            turning = turn_start <= i < turn_start + turn_len
            # occasional GPS multipath noise burst on the reported position
            burst = 6.0 if rng.random() < 0.07 else 1.0
            nlat = GPS_POS_NOISE_M * burst / 111_320.0
            nlon = GPS_POS_NOISE_M * burst / (111_320.0 * np.cos(np.radians(lat)))
            rows.append({
                "track_id": track_id,
                "callsign": callsign,
                "t": float(i * dt),
                "lat": lat + rng.normal(0, nlat),
                "lon": lon + rng.normal(0, nlon),
                "alt": alt + rng.normal(0, 8.0),
                "velocity": v + rng.normal(0, 1.5),
                "heading": (heading + rng.normal(0, HEADING_NOISE_DEG)) % 360,
                "vertical_rate": (vrate if turning else 0.0) + rng.normal(0, 0.5),
                "is_attack": False,
                "attack_type": "none",
            })
            # advance - apply the turn/climb during the manoeuvre window
            step = turn_rate if turning else 0.0
            heading = (heading + step + rng.normal(0, HEADING_NOISE_DEG)) % 360
            if turning:
                alt = float(np.clip(alt + vrate * dt, 6000, 12500))
            lat, lon = destination_point(lat, lon, heading, v * dt / 1000.0)

    return pd.DataFrame(rows)


def generate_hotspot_traffic(jlat: float, jlon: float,
                             n_aircraft: int = 12,
                             samples: int = 30,
                             dt: float = config.DETECT.sample_dt_s,
                             dist_band_km: tuple[float, float] = (30.0, 260.0),
                             seed: int = 31) -> pd.DataFrame:
    """Traffic on inbound courses toward a point (e.g. a jammer).

    Aircraft start on a ring around the point and fly toward it. Within the sim
    window the near ones cross into the interference radius (becoming zone
    evidence) while the far ones are still approaching (so they receive
    pre-emptive ground->air alerts). Same schema as `generate_baseline`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_aircraft):
        d0 = rng.uniform(*dist_band_km)
        brg_from_jammer = rng.uniform(0, 360)
        lat, lon = destination_point(jlat, jlon, brg_from_jammer, d0)
        heading = bearing_deg(lat, lon, jlat, jlon)  # inbound
        v = rng.uniform(*CRUISE_SPEED_MS)
        alt = rng.uniform(*CRUISE_ALT_M)
        track_id, callsign = f"hot{k:04d}", f"HOT{k:03d}"
        for i in range(samples):
            nlat = GPS_POS_NOISE_M / 111_320.0
            nlon = GPS_POS_NOISE_M / (111_320.0 * np.cos(np.radians(lat)))
            rows.append({
                "track_id": track_id, "callsign": callsign, "t": float(i * dt),
                "lat": lat + rng.normal(0, nlat), "lon": lon + rng.normal(0, nlon),
                "alt": alt + rng.normal(0, 8.0), "velocity": v + rng.normal(0, 1.5),
                "heading": (heading + rng.normal(0, HEADING_NOISE_DEG)) % 360,
                "vertical_rate": rng.normal(0, 0.5),
                "is_attack": False, "attack_type": "none",
            })
            heading = (heading + rng.normal(0, HEADING_NOISE_DEG)) % 360
            lat, lon = destination_point(lat, lon, heading, v * dt / 1000.0)
    return pd.DataFrame(rows)


def build_scenario(region: config.Region = config.DEFAULT_REGION,
                   n_background: int = 30,
                   n_hotspot: int = 14,
                   samples: int = 30,
                   jammer: tuple[float, float] = (41.3, 65.6),
                   jammer_radius_km: float = 45.0,
                   scatter_fraction: float = 0.22,
                   seed: int = 7):
    """Compose a full corridor scenario: honest background traffic, a few scattered
    spoof/jam attacks (for per-signature recall), and one ground-jammer hotspot
    that produces a real interference zone.

    Returns (tracks, truth, jammers) where jammers is a list of dicts.
    """
    from src.simulate.attacks import inject_attacks, apply_jammer

    jlat, jlon = jammer
    bg = generate_baseline(n_aircraft=n_background, samples=samples,
                           region=region, seed=seed)
    bg, truth_bg = inject_attacks(bg, fraction=scatter_fraction, seed=seed + 4)

    hot = generate_hotspot_traffic(jlat, jlon, n_aircraft=n_hotspot,
                                   samples=samples, seed=seed + 9)
    hot, truth_hot = apply_jammer(hot, jlat, jlon, radius_km=jammer_radius_km,
                                  seed=seed + 13)

    tracks = pd.concat([bg, hot], ignore_index=True)
    truth = pd.concat([truth_bg, truth_hot], ignore_index=True)
    jammers = [{"lat": jlat, "lon": jlon, "radius_km": jammer_radius_km}]
    return tracks, truth, jammers


if __name__ == "__main__":
    tr, tru, jam = build_scenario()
    print(f"{tr['track_id'].nunique()} tracks, {len(tr)} samples")
    print(tru["attack_type"].value_counts())
