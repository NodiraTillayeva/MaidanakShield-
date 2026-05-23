"""
End-to-end orchestration for Maidanak Sentinel.

Two entry points:

  run_testbench()  -> generate labelled traffic, run the full two-layer detector,
                      build zones + alerts, and score performance. This is what the
                      validation figures and the dashboard's "evidence" tab use.

  live_overlay()   -> pull a real OpenSky snapshot and the real GPS sky so the
                      dashboard can show the system running on current open data.

Keeping it here means the Streamlit app and the figure exporter run *identical*
logic - the demo and the evidence can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

import config
from src.detect import physics_twin, rf_shield, optical_verify, integrity
from src.alert import preemptive
from src.eval import metrics
from src.simulate.scenario import build_scenario
from src.ingest import opensky, gnss, adsb


@dataclass
class TestbenchResult:
    tracks: pd.DataFrame          # labelled long-format tracks (with injected attacks)
    scored: pd.DataFrame          # per-sample physics cues + score
    track_scores: pd.DataFrame    # per-aircraft fused score (physics + optical)
    truth: pd.DataFrame           # ground-truth labels
    zones: pd.DataFrame           # detected interference zones
    alerts: pd.DataFrame          # pre-emptive ground->air alerts
    jammers: list                 # ground-truth jammer locations
    sky: dict                     # real GPS sky used for the optical cue
    metrics: dict                 # ROC/precision/recall
    per_attack: pd.DataFrame      # recall by attack type


def run_testbench(n_background: int = 30,
                  n_hotspot: int = 14,
                  samples: int = 30,
                  region: config.Region = config.DEFAULT_REGION,
                  when: datetime | None = None,
                  use_cache: bool = True,
                  seed: int = 7) -> TestbenchResult:
    """OPTIONAL offline stress-test of the Sentinel Twin against *simulated* attacks.

    This is NOT part of the live demo, dashboard, or slide deck - those run purely
    on real data (see `run_live_detection`). It exists only to measure detection
    limits with ground-truth labels, which real data cannot provide cleanly.
    """
    when = when or datetime.now(timezone.utc)

    tracks, truth, jammers = build_scenario(region=region, n_background=n_background,
                                            n_hotspot=n_hotspot, samples=samples,
                                            seed=seed)

    scored = physics_twin.score_all(tracks)               # Sentinel Twin
    summary = physics_twin.track_summary(scored)
    track_scores, sky = optical_verify.apply_optical_cue( # Maidanak Observatory
        summary, region=region, when=when, use_cache=use_cache)

    zones = rf_shield.interference_zones(scored)          # Maidanak Shield
    states = preemptive.current_states_from_tracks(tracks)
    # only pre-warn aircraft that are NOT already inside an attack (the honest
    # inbound traffic) - that is the population the handoff is designed to protect.
    flagged_ids = set(track_scores.loc[track_scores["flagged"], "track_id"])
    clean_states = states[~states["track_id"].isin(flagged_ids)]
    alerts = preemptive.generate_alerts(clean_states, zones)

    mt = metrics.evaluate(track_scores, truth)
    pa = metrics.per_attack_recall(track_scores, truth)

    return TestbenchResult(tracks=tracks, scored=scored, track_scores=track_scores,
                           truth=truth, zones=zones, alerts=alerts, jammers=jammers,
                           sky=sky, metrics=mt, per_attack=pa)


@dataclass
class LiveResult:
    region: config.WatchRegion
    aircraft: pd.DataFrame        # real, classified aircraft (integrity columns)
    zones: pd.DataFrame          # real interference zones (degraded clusters)
    alerts: pd.DataFrame         # healthy aircraft projected into a zone
    sky: dict                    # real GPS constellation over the worst zone
    summary: dict                # headline real stats
    cross_feed: dict             # independent second-feed agreement
    fetched_at: str


def run_live_detection(region_key: str = config.DEFAULT_WATCH,
                       use_cache: bool = False,
                       cross_check: bool = True) -> LiveResult:
    """Full real-data detection over one watch region - no simulation.

    Maidanak Shield  : classify real NIC/NACp -> flag degraded aircraft -> zones.
    Observatory      : real GPS constellation over the worst zone confirms the sky
                       is healthy, so degradation is interference, not geometry.
    Pre-emptive alert: healthy aircraft on course into a zone are pre-warned.
    Validation       : an independent second ADS-B feed reproduces the footprint.
    """
    wr = config.WATCH_REGIONS[region_key]
    raw = adsb.fetch_region(wr.lat, wr.lon, wr.dist_nm, use_cache=use_cache, cache_tag=region_key)
    ac = integrity.classify(raw)
    air = ac[ac["airborne"]].copy() if not ac.empty else ac

    # real jamming footprints are large, so cluster on a coarser grid (~130 km)
    zones = (rf_shield.zones_from_points(air, grid_deg=1.2, min_count=2)
             if not air.empty else pd.DataFrame())
    summ = integrity.summary(ac)

    # observatory confirmation over the worst zone (or region centre)
    if not zones.empty:
        z0 = zones.iloc[0]
        sky = optical_verify.confirm_zone(float(z0["lat"]), float(z0["lon"]), use_cache=use_cache)
    else:
        sky = optical_verify.confirm_zone(wr.lat, wr.lon, use_cache=use_cache)

    # pre-emptive alerts: only healthy aircraft (the ones still worth warning)
    alerts = pd.DataFrame()
    if not zones.empty and not air.empty:
        healthy = air[~air["flagged"]][["track_id", "callsign", "lat", "lon",
                                        "heading", "velocity"]].copy()
        healthy["velocity"] = healthy["velocity"].fillna(230.0)
        alerts = preemptive.generate_alerts(healthy, zones)

    # independent cross-feed reproducibility check
    cross = {}
    if cross_check and not use_cache:
        try:
            other = adsb.fetch_region(wr.lat, wr.lon, wr.dist_nm, feed="airplanes.live",
                                      cache_tag=f"{region_key}_xcheck")
            oc = integrity.classify(other)
            os_ = integrity.summary(oc)
            cross = {"feed": "airplanes.live", "aircraft": os_["aircraft"],
                     "degraded": os_["degraded"], "pct_degraded": os_["pct_degraded"]}
        except Exception:  # noqa: BLE001
            cross = {}

    return LiveResult(region=wr, aircraft=ac, zones=zones, alerts=alerts, sky=sky,
                      summary=summ, cross_feed=cross,
                      fetched_at=datetime.now(timezone.utc).isoformat())


def scan_live(use_cache: bool = False) -> pd.DataFrame:
    """Live interference league table across all watch regions."""
    return adsb.scan_watch_regions(use_cache=use_cache)


def live_overlay(region: config.Region = config.DEFAULT_REGION,
                 use_cache: bool = False) -> dict:
    """Real, current open data: live aircraft snapshot + real GPS sky over Maidanak.

    OpenSky's crowd-sourced ground-receiver coverage over the Central Asian
    corridor is genuinely sparse (often zero aircraft in the tight box) - itself a
    symptom of the problem this project addresses. So if the corridor box is empty
    we widen the search and flag the result as `widened`, but still real data.
    """
    snap = opensky.fetch_snapshot(region, use_cache=use_cache)
    widened = False
    if snap.empty:
        wide = config.Region("wide", "Wider Central Asia", region.lamin - 10,
                             region.lomin - 14, region.lamax + 8, region.lomax + 12)
        snap = opensky.fetch_snapshot(wide, use_cache=False)
        widened = not snap.empty

    obs = config.MAIDANAK_OBSERVATORY
    sky = optical_verify.sky_health(obs["lat"], obs["lon"],
                                    elevation_m=obs["elevation_m"], use_cache=use_cache)
    return {"snapshot": snap, "observatory_sky": sky, "widened": widened,
            "n_aircraft": int(len(snap)),
            "fetched_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    r = run_testbench(use_cache=True)
    m = r.metrics
    print(f"Aircraft: {m['n']} | attacks: {m['n_attacks']}")
    print(f"AUC={m.get('auc'):.3f}  precision={m.get('precision'):.3f}  "
          f"recall={m.get('recall'):.3f}  F1={m.get('f1'):.3f}")
    print(f"Interference zones detected: {len(r.zones)}")
    print(f"Pre-emptive alerts raised: {len(r.alerts)}")
    print("Recall by attack type:")
    print(r.per_attack.to_string(index=False))
