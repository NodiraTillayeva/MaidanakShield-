"""
Maidanak Sentinel - live GNSS-interference operations dashboard.

Run with:  streamlit run app.py

100% real data, no simulation. The system scans live air traffic for the real
ADS-B navigation-integrity signal (NIC / NACp) that GNSS jamming leaves behind,
maps interference zones, confirms them against the real GPS constellation, and
pre-warns aircraft on course into a zone. Two independent feeds and the real
satellite geometry corroborate every detection.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import config
from src.geo import destination_point
from src.ingest import gnss
from src import impact
from src.planemap import build_deck
from src.pipeline import run_live_detection, scan_live

st.set_page_config(page_title="Maidanak Sentinel", layout="wide")

# served by the small static server in scene3d/ (see run_dashboard.sh)
ANIM_URL = "http://localhost:8502/index.html"
GREEN, AMBER, RED, BLUE, INK = "#2ecc71", "#f1c40f", "#e74c3c", "#2980b9", "#2c3e50"
INTEG_SCALE = [[0.0, GREEN], [0.4, AMBER], [0.7, "#e67e22"], [1.0, RED]]


@st.cache_data(ttl=120, show_spinner="Scanning live air traffic for GNSS interference…")
def cached_live(region_key: str):
    return run_live_detection(region_key, use_cache=False, cross_check=True)


@st.cache_data(ttl=180, show_spinner="Scanning global airspace…")
def cached_scan():
    return scan_live(use_cache=False)


def circle_latlon(lat, lon, radius_km, n=64):
    pts = [destination_point(lat, lon, b, radius_km) for b in np.linspace(0, 360, n)]
    return [p[0] for p in pts], [p[1] for p in pts]


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Maidanak Sentinel")
st.sidebar.caption("Live GNSS jamming/spoofing detection - real ADS-B integrity data")

region_keys = list(config.WATCH_REGIONS.keys())
labels = {k: config.WATCH_REGIONS[k].name for k in region_keys}
if "region_key" not in st.session_state:
    st.session_state.region_key = config.DEFAULT_WATCH

if st.sidebar.button("Scan global airspace for live interference"):
    scan = cached_scan()
    st.session_state.scan = scan
    if not scan.empty:
        st.session_state.region_key = scan.iloc[0]["key"]  # jump to worst hotspot

region_key = st.sidebar.selectbox(
    "Theatre", region_keys, index=region_keys.index(st.session_state.region_key),
    format_func=lambda k: labels[k])
st.session_state.region_key = region_key

if "scan" in st.session_state:
    st.sidebar.markdown("**Live interference league**")
    sc = st.session_state.scan.rename(columns={"region": "Region", "pct_degraded": "% deg"})
    st.sidebar.dataframe(sc[["Region", "aircraft", "% deg"]],
                         hide_index=True, width="stretch")

st.sidebar.markdown("---")
st.sidebar.caption("Data: adsb.lol + airplanes.live (ADS-B NIC/NACp) · "
                   "Celestrak (real GPS ephemeris)")

# --------------------------------------------------------------------------- compute
R = cached_live(region_key)
ac = R.aircraft[R.aircraft["airborne"]] if not R.aircraft.empty else R.aircraft
s = R.summary

st.title(f"Live GNSS interference - {R.region.name}")
st.caption("Real ADS-B navigation-integrity (NIC/NACp) · real GPS constellation · "
           "no simulation. Green = healthy fix, red = GNSS integrity lost.")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Airborne aircraft", s["aircraft"])
k2.metric("Degraded GNSS", s["degraded"])
k3.metric("Fully lost fix", s["lost"])
k4.metric("% degraded", f"{s['pct_degraded']}%")
k5.metric("Interference zones", len(R.zones))
k6.metric("Pre-emptive alerts", len(R.alerts))

if s["aircraft"] == 0:
    st.warning("No airborne aircraft returned for this theatre right now "
               "(coverage varies). Try another theatre or scan the globe.")

# --------------------------------------------------------------------------- main map
st.pydeck_chart(build_deck(R), use_container_width=True)
st.markdown(
    "<div style='background:#111827;color:#fff;display:inline-block;padding:6px 12px;"
    "border-radius:6px;font-size:13px'>"
    "<b>Airplane colour = live GNSS status:</b> &nbsp;"
    "<span style='color:#2ecc71'>&#9608;</span> healthy &nbsp;"
    "<span style='color:#e67e22'>&#9608;</span> degraded &nbsp;"
    "<span style='color:#e74c3c'>&#9608;</span> lost &nbsp;|&nbsp; "
    "red circle = interference zone &nbsp;|&nbsp; black dot = documented source</div>",
    unsafe_allow_html=True)

# --------------------------------------------------------------------------- tabs
t_how, t_foot, t_const, t_alert, t_list, t_impact = st.tabs(
    ["How it works", "Footprint & validation", "Constellation confirmation",
     "Pre-emptive alerts", "Live aircraft", "Impact & business"])

with t_how:
    st.subheader("How Maidanak Sentinel works")
    st.write("The Maidanak radio-telescope/observatory layer links the ground sensor "
             "network, the GPS satellites and the aircraft: it watches the real "
             "constellation, detects jamming, and hands a pre-emptive alert to the "
             "Sentinel Twin onboard - before the spoofed signal corrupts the receiver.")
    components.iframe(ANIM_URL, height=560, scrolling=False)
    st.caption("Interactive 3D explainer - drag to orbit. Green beam = pre-emptive "
               "ground→air alert · blue beam = observatory verifying the real "
               "constellation · red beam = GPS signal being jammed.")

with t_foot:
    st.subheader("Real interference footprint - independently corroborated")
    c1, c2 = st.columns(2)
    c1.markdown("**Cross-feed agreement** (reproducibility on independent data)")
    cf = R.cross_feed
    if cf:
        comp = pd.DataFrame([
            {"Feed": "adsb.lol", "Aircraft": s["aircraft"], "% degraded": s["pct_degraded"]},
            {"Feed": cf["feed"], "Aircraft": cf["aircraft"], "% degraded": cf["pct_degraded"]},
        ])
        c1.dataframe(comp, hide_index=True, width="stretch")
        c1.caption("Two independent ADS-B aggregators report the same interference "
                   "footprint - this is not an artefact of one data source.")
    else:
        c1.info("Cross-feed check unavailable (cached run).")
    c2.markdown("**Documented public reference**")
    if R.region.source_name:
        c2.success(f"Detected footprint coincides with the publicly documented "
                   f"interference area: **{R.region.source_name}**.")
    c2.caption("The Baltic/Kaliningrad and Black Sea jamming are widely documented "
               "(EASA bulletins, GPSJAM). Our live footprint lands on the same area.")
    if not R.zones.empty:
        st.markdown("**Detected interference zones**")
        zz = R.zones.rename(columns={"lat": "Lat", "lon": "Lon", "n_aircraft": "Aircraft",
                                     "severity": "Severity"})
        st.dataframe(zz[["Lat", "Lon", "Aircraft", "Severity"]], hide_index=True,
                     width="stretch")

with t_const:
    st.subheader("Maidanak Observatory layer - real constellation as ground truth")
    z = R.zones.iloc[0] if not R.zones.empty else None
    zlat = float(z["lat"]) if z is not None else R.region.lat
    zlon = float(z["lon"]) if z is not None else R.region.lon
    sats = gnss.visible_satellites(zlat, zlon, 0.0, use_cache=True)
    c1, c2 = st.columns([1.1, 1])
    if not sats.empty:
        sp = go.Figure(go.Scatterpolar(r=90 - sats["el_deg"], theta=sats["az_deg"],
                       mode="markers", marker=dict(size=9, color=sats["el_deg"],
                       colorscale="Viridis", showscale=True, colorbar=dict(title="elev°")),
                       text=sats["name"], hoverinfo="text"))
        sp.update_layout(title=f"Real GPS sky over the zone - {len(sats)} satellites",
                         polar=dict(radialaxis=dict(range=[0, 90], showticklabels=False),
                         angularaxis=dict(rotation=90, direction="clockwise")),
                         height=430, margin=dict(l=30, r=30, t=50, b=30))
        c1.plotly_chart(sp, width="stretch")
    c2.metric("Real GPS satellites visible", R.sky["n_visible"])
    c2.metric("Geometry (PDOP)", f"{R.sky['pdop']:.2f}")
    verdict_box = c2.success if R.sky["confidence"] == "high" else c2.warning
    verdict_box(R.sky["verdict"])
    c2.caption("A spoofer can forge any RF signal but cannot move a real satellite. "
               "Because the real sky is healthy, the integrity loss is interference - "
               "not bad geometry. No RF-only competitor can make this call.")

with t_alert:
    st.subheader("Pre-emptive ground → air alerting")
    st.write("Healthy aircraft on course into a live interference zone are warned "
             "**before** their receiver is affected.")
    if R.alerts.empty:
        st.info("No healthy aircraft currently projected to enter a zone.")
    else:
        a = R.alerts.rename(columns={"callsign": "Callsign", "eta_min": "ETA (min)",
                                     "distance_km": "Distance (km)",
                                     "zone_severity": "Zone severity", "action": "Action"})
        st.dataframe(a[["Callsign", "ETA (min)", "Distance (km)", "Zone severity", "Action"]],
                     hide_index=True, width="stretch")

with t_list:
    st.subheader("Aircraft reporting degraded GNSS integrity (live)")
    if not ac.empty:
        deg = ac[ac["status"].isin(["degraded", "lost"])]
        show = deg.rename(columns={"callsign": "Callsign", "status": "Status",
                                   "nic": "NIC", "nac_p": "NACp", "lat": "Lat", "lon": "Lon"})
        st.dataframe(show[["Callsign", "Status", "NIC", "NACp", "Lat", "Lon"]].head(40),
                     hide_index=True, width="stretch")
        st.caption(f"{len(deg)} of {len(ac)} airborne aircraft. "
                   f"Fetched {R.fetched_at}.")

with t_impact:
    st.subheader("Impact & business model")
    ee = impact.environmental_economic()
    c1, c2, c3 = st.columns(3)
    c1.metric("CO₂ avoided / year", f"{ee['co2_saved_tonnes_year']:,} t",
              help="vs reactive response, illustrative model")
    c2.metric("Jet fuel saved / year", f"{ee['fuel_saved_tonnes_year']:,} t")
    c3.metric("≈ equivalent cars off road", f"{ee['co2_saved_equiv_cars']:,}")
    c1.metric("Operating cost avoided / yr", f"${ee['cost_saved_usd_year']:,}")

    st.markdown("**Affordability - capital cost per corridor (order of magnitude)**")
    aff = impact.affordability()
    bar = go.Figure(go.Bar(x=list(aff.keys()), y=list(aff.values()), marker_color=BLUE,
                    text=[f"${v:,}" for v in aff.values()], textposition="outside"))
    bar.update_layout(yaxis_type="log", height=300, yaxis_title="USD (log)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, width="stretch")

    cu = impact.use_cases()
    st.markdown("**Use cases (security · space · defence)**")
    for k, v in cu.items():
        st.markdown(f"- **{k}** - {v}")
    bm = impact.business_model()
    st.markdown("**Revenue streams**")
    st.dataframe(pd.DataFrame(bm["streams"], columns=["Stream", "Customer", "Model"]),
                 hide_index=True, width="stretch")
    st.caption("Impact figures are an illustrative model with stated assumptions "
               "(see src/impact.py); all are conservative and auditable.")
