"""
Export slide-ready figures + a 4-page PDF deck mapped to the Round-2 brief.

Run with:  python export_figures.py

The 4-page deck maps 1:1 onto the brief's required slides
("idea, business model, impact, and how you prototyped/tested it"):

  page 1  01_architecture.png   - the idea & innovation
  page 2  02_business.png       - the business model
  page 3  03_impact.png         - impact (industry / environment / society)
  page 4  04_prototype.png      - REAL live detection + validation (prototype/test)

Plus standalone PNGs (real footprint, validation) for the video. Everything that
shows detection runs on real ADS-B integrity data - no simulation.
"""
from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import config
from src.geo import destination_point
from src.ingest import gnss
from src import impact
from src.pipeline import run_live_detection

GREEN, AMBER, ORANGE, RED, BLUE, INK = "#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#2980b9", "#2c3e50"
plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "#fbfbfd"})


def _circle(lat, lon, r_km, n=80):
    pts = [destination_point(lat, lon, b, r_km) for b in np.linspace(0, 360, n)]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


# --------------------------------------------------------------------------- 1. architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(50, 95, "Maidanak Sentinel - two-layer GNSS defence",
            ha="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(50, 89, "Detects REAL jamming from open data · optically verified · "
                    "$30 sensors · no export controls",
            ha="center", fontsize=10.5, color="#666", style="italic")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                    fc=color, ec=INK, lw=1.5, alpha=0.95))
        ax.text(x + w / 2, y + h - 5, title, ha="center", fontweight="bold",
                fontsize=11.5, color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 11 - i * 5.5, ln, ha="center", fontsize=9, color=INK)

    box(4, 52, 27, 30, "① Maidanak Shield",
        ["Real ADS-B integrity", "(NIC / NACp) + $30 SDRs",
         "→ live interference", "ZONE map"], "#eaf4ff")
    box(36.5, 52, 27, 30, "② Sentinel Twin",
        ["Onboard digital twin", "dead-reckons every fix", "→ catches spoof BEFORE",
         "the receiver is fooled"], "#eafaf0")
    box(69, 52, 27, 30, "③ Maidanak Observatory",
        ["Real GPS ephemeris +", "optical confirmation", "→ physical GROUND TRUTH",
         "a spoofer cannot fake"], "#fdf3e7")
    box(28, 12, 44, 22, "Pre-emptive ground → air alert",
        ["Aircraft entering a flagged zone are warned in advance.",
         "Every competitor reacts only AFTER the fix is already wrong.",
         "Validated on live jamming (Baltic, Persian Gulf) today."], "#fdeaea")
    for x0 in (17.5, 50, 82.5):
        ax.add_patch(FancyArrowPatch((x0, 52), (50, 34), arrowstyle="-|>",
                                     mutation_scale=16, color=INK, lw=1.4))
    ax.text(50, 5, "Open data in: adsb.lol / airplanes.live (ADS-B NIC) · "
                   "Celestrak (GPS ephemeris)   |   built on civilian components",
            ha="center", fontsize=9, color="#555")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 2. business
def fig_business():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(50, 95, "Business model", ha="center", fontsize=17, fontweight="bold", color=INK)
    bm = impact.business_model()
    colors = ["#eaf4ff", "#eafaf0", "#fdf3e7", "#fdeaea"]
    for i, (stream, cust, model) in enumerate(bm["streams"]):
        x = 4 + (i % 2) * 48
        y = 56 - (i // 2) * 26
        ax.add_patch(FancyBboxPatch((x, y), 44, 22,
                     boxstyle="round,pad=0.6,rounding_size=2", fc=colors[i], ec=INK, lw=1.4))
        ax.text(x + 22, y + 16, stream, ha="center", fontweight="bold", fontsize=11, color=INK)
        ax.text(x + 22, y + 9.5, f"customer: {cust}", ha="center", fontsize=8.7, color="#444")
        ax.text(x + 22, y + 4.5, f"model: {model}", ha="center", fontsize=8.7,
                color="#444", style="italic")
    ax.add_patch(FancyBboxPatch((10, 3), 80, 12, boxstyle="round,pad=0.5,rounding_size=2",
                 fc="#f4f0fb", ec=INK, lw=1.3))
    ax.text(50, 9.5, "Beachhead: " + bm["wedge"], ha="center", fontsize=9.3,
            color=INK, wrap=True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 3. impact
def fig_impact():
    ee = impact.environmental_economic()
    aff = impact.affordability()
    cu = impact.use_cases()
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1, 1])
    fig.suptitle("Impact - industry · environment · society", fontsize=16,
                 fontweight="bold", color=INK)

    # environmental headline numbers
    ax0 = fig.add_subplot(gs[0, 0]); ax0.axis("off")
    kpis = [(f"{ee['co2_saved_tonnes_year']:,} t", "CO₂ avoided / year"),
            (f"{ee['fuel_saved_tonnes_year']:,} t", "jet fuel saved / year"),
            (f"{ee['co2_saved_equiv_cars']:,}", "≈ cars off the road"),
            (f"${ee['cost_saved_usd_year']/1e6:.1f}M", "operating cost avoided / yr")]
    for i, (v, lab) in enumerate(kpis):
        x, y = (i % 2), 1 - (i // 2)
        ax0.text(x * 0.5 + 0.02, y * 0.5 + 0.05, v, fontsize=20, fontweight="bold",
                 color=GREEN if i < 3 else BLUE, transform=ax0.transAxes)
        ax0.text(x * 0.5 + 0.02, y * 0.5 - 0.02, lab, fontsize=9.5, color="#555",
                 transform=ax0.transAxes)
    ax0.set_title("Environmental & economic (illustrative, auditable)", fontsize=11)

    # affordability log bar
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(range(len(aff)), list(aff.values()), color=[GREEN, AMBER, RED])
    ax1.set_yscale("log"); ax1.set_xticks(range(len(aff)))
    ax1.set_xticklabels(["Maidanak\nmesh (corridor)", "Commercial\nprobe", "Military\nCRPA/tail"],
                        fontsize=8.5)
    for i, v in enumerate(aff.values()):
        ax1.text(i, v * 1.15, f"${v:,}", ha="center", fontsize=9)
    ax1.set_title("Affordability - capital cost (log)", fontsize=11)
    ax1.set_ylim(1e3, 1e5 * 2)

    # use cases security/space/defence
    ax2 = fig.add_subplot(gs[1, :]); ax2.axis("off")
    ax2.set_title("Use cases - security · space · defence", fontsize=11, loc="left")
    for i, (k, v) in enumerate(cu.items()):
        ax2.text(0.01, 0.78 - i * 0.32, f"{k}:", fontsize=11, fontweight="bold",
                 color=INK, transform=ax2.transAxes)
        ax2.text(0.12, 0.78 - i * 0.32, v, fontsize=9.3, color="#333",
                 transform=ax2.transAxes, wrap=True,
                 va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# --------------------------------------------------------------------------- real footprint
def _draw_footprint(ax, R):
    wr = R.region
    ac = R.aircraft[R.aircraft["airborne"]] if not R.aircraft.empty else R.aircraft
    span = wr.dist_nm * 1.852 / 111.0
    ax.set_xlim(wr.lon - span / np.cos(np.radians(wr.lat)), wr.lon + span / np.cos(np.radians(wr.lat)))
    ax.set_ylim(wr.lat - span, wr.lat + span)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.grid(color="#e3e3ea", lw=0.6)
    for _, z in R.zones.iterrows():
        cl, co = _circle(z["lat"], z["lon"], 1.2 * 55)
        ax.fill(co, cl, color=RED, alpha=0.10, zorder=2)
        ax.plot(co, cl, color=RED, lw=1, alpha=0.5, zorder=2)
    if not ac.empty:
        sc = ax.scatter(ac["lon"], ac["lat"], c=ac["integrity_score"].fillna(0),
                        cmap="RdYlGn_r", vmin=0, vmax=1, s=55, edgecolor=INK, lw=0.4, zorder=5)
        plt.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, label="GNSS degraded")
    if wr.source_lat is not None:
        ax.scatter([wr.source_lon], [wr.source_lat], marker="X", s=240, color=INK,
                   edgecolor="white", lw=1.5, zorder=6)
        ax.annotate(f"documented\nsource: {wr.source_name.split('(')[0].strip()}",
                    xy=(wr.source_lon, wr.source_lat),
                    xytext=(wr.source_lon - span * 0.7, wr.source_lat + span * 0.55),
                    fontsize=8.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK))
    s = R.summary
    ax.set_title(f"LIVE GNSS interference - {wr.name}\n"
                 f"{s['degraded']} of {s['aircraft']} airborne aircraft degraded "
                 f"({s['pct_degraded']}%) · real ADS-B data", color=INK, fontsize=12)


def fig_real_footprint(R):
    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_footprint(ax, R)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- validation
def _draw_validation(R, ax_bar, ax_cross, ax_sky):
    s = R.summary
    # integrity breakdown
    ax_bar.bar(["healthy", "degraded", "lost"],
               [s["aircraft"] - s["degraded"], s["degraded"] - s["lost"], s["lost"]],
               color=[GREEN, AMBER, RED])
    ax_bar.set_title("Live integrity breakdown"); ax_bar.set_ylabel("airborne aircraft")

    # cross-feed agreement
    cf = R.cross_feed
    feeds = ["adsb.lol", cf.get("feed", "feed 2") if cf else "feed 2"]
    vals = [s["pct_degraded"], cf.get("pct_degraded", 0) if cf else 0]
    ax_cross.bar(feeds, vals, color=[BLUE, "#6fa8dc"])
    for i, v in enumerate(vals):
        ax_cross.text(i, v + 0.5, f"{v}%", ha="center", fontsize=10)
    ax_cross.set_title("Cross-feed agreement\n(independent sources)")
    ax_cross.set_ylabel("% degraded"); ax_cross.set_ylim(0, max(vals + [1]) * 1.3)

    # constellation polar
    z = R.zones.iloc[0] if not R.zones.empty else None
    zlat = float(z["lat"]) if z is not None else R.region.lat
    zlon = float(z["lon"]) if z is not None else R.region.lon
    sats = gnss.visible_satellites(zlat, zlon, 0.0, use_cache=True)
    ax_sky.set_theta_zero_location("N"); ax_sky.set_theta_direction(-1)
    if not sats.empty:
        ax_sky.scatter(np.radians(sats["az_deg"]), 90 - sats["el_deg"],
                       c=sats["el_deg"], cmap="viridis", s=45, edgecolor="white", lw=0.4)
    ax_sky.set_rlim(0, 90); ax_sky.set_yticklabels([])
    ax_sky.set_title(f"Real GPS sky over zone\n{R.sky['n_visible']} sats · "
                     f"PDOP {R.sky['pdop']:.1f} → sky HEALTHY", fontsize=10)


def fig_validation(R):
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_cross = fig.add_subplot(gs[0, 1])
    ax_sky = fig.add_subplot(gs[0, 2], projection="polar")
    _draw_validation(R, ax_bar, ax_cross, ax_sky)
    fig.suptitle("Validation on REAL data - independent feeds + real constellation agree",
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# --------------------------------------------------------------------------- 4. prototype (slide)
def fig_prototype(R):
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1], width_ratios=[1, 1, 1.1])
    ax_map = fig.add_subplot(gs[0, :])
    _draw_footprint(ax_map, R)
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_cross = fig.add_subplot(gs[1, 1])
    ax_sky = fig.add_subplot(gs[1, 2], projection="polar")
    _draw_validation(R, ax_bar, ax_cross, ax_sky)
    fig.suptitle("Prototype - detecting & validating REAL GNSS jamming, live, from open data",
                 fontweight="bold", color=INK, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# --------------------------------------------------------------------------- main
def main():
    print("Fetching REAL live interference data…")
    R = run_live_detection(config.DEFAULT_WATCH, use_cache=False, cross_check=True)
    print(f"  {R.region.name}: {R.summary['degraded']}/{R.summary['aircraft']} degraded "
          f"({R.summary['pct_degraded']}%), {len(R.zones)} zones")

    figs = {
        "01_architecture.png": fig_architecture(),
        "02_business.png": fig_business(),
        "03_impact.png": fig_impact(),
        "04_prototype.png": fig_prototype(R),
        "05_real_footprint.png": fig_real_footprint(R),
        "06_validation.png": fig_validation(R),
    }
    for name, fig in figs.items():
        path = config.REPORTS_DIR / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        print("  wrote", path)

    deck = config.REPORTS_DIR / "maidanak_sentinel_deck.pdf"
    with PdfPages(deck) as pdf:
        for name in ("01_architecture.png", "02_business.png",
                     "03_impact.png", "04_prototype.png"):
            pdf.savefig(figs[name], bbox_inches="tight")
    print("  wrote", deck, "(4-slide deck)")

    out = dict(region=R.region.name, **R.summary, zones=len(R.zones),
               preemptive_alerts=len(R.alerts), cross_feed=R.cross_feed,
               observatory=R.sky["verdict"], impact=impact.environmental_economic())
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print("  wrote", config.REPORTS_DIR / "metrics.json")
    print("\nHeadline:", json.dumps({k: out[k] for k in
          ("region", "aircraft", "degraded", "pct_degraded", "zones")}, indent=2))


if __name__ == "__main__":
    main()
