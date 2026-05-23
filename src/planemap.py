"""
Main live map with rotated airplane icons (pydeck / deck.gl IconLayer).

Each aircraft is drawn as an airplane glyph rotated to its heading and tinted by
GNSS-integrity status (green healthy, orange degraded, red lost). Interference
zones and the documented source are overlaid. Token-free Carto basemap.
"""
from __future__ import annotations

import base64
import io

import pandas as pd
import pydeck as pdk

import config

STATUS_RGB = {"healthy": [39, 174, 96], "degraded": [230, 126, 34],
              "lost": [231, 76, 60], "unknown": [140, 140, 140]}

_ICON_URL = None
_ICON_PX = 144


def airplane_icon_url() -> str:
    """White top-down airplane silhouette as a base64 PNG data-URI (cached).

    Drawn as a mask so deck.gl's IconLayer can tint it per-aircraft by status.
    """
    global _ICON_URL
    if _ICON_URL:
        return _ICON_URL
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(_ICON_PX / 96, _ICON_PX / 96), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # unicode airplane glyph, rotated so the nose points up (north) at angle 0
    ax.text(0.5, 0.5, "✈", fontsize=78, ha="center", va="center",
            color="white", rotation=-45)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True); plt.close(fig)
    _ICON_URL = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return _ICON_URL


def build_deck(R, zoom: float = 4.2):
    """Return a pydeck.Deck of the live theatre with airplane icons."""
    wr = R.region
    # per-row icon dict (the data: URL lives in row data, not in a top-level prop
    # that deck.gl would try to parse as an accessor expression)
    icon = {"url": airplane_icon_url(), "width": _ICON_PX, "height": _ICON_PX,
            "anchorX": _ICON_PX // 2, "anchorY": _ICON_PX // 2, "mask": True}

    ac = R.aircraft[R.aircraft["airborne"]] if not R.aircraft.empty else pd.DataFrame()
    rows = []
    for _, a in ac.iterrows():
        rows.append({
            "lon": float(a["lon"]), "lat": float(a["lat"]),
            "angle": -float(a.get("heading") or 0.0),   # deck angle is CCW; heading is CW
            "color": STATUS_RGB.get(a.get("status"), STATUS_RGB["unknown"]),
            "icon_data": icon,
            "callsign": str(a.get("callsign", "")),
            "status": str(a.get("status", "")),
            "nic": "" if pd.isna(a.get("nic")) else int(a.get("nic")),
        })
    adf = pd.DataFrame(rows)

    layers = []
    if not R.zones.empty:
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=R.zones.copy(), get_position="[lon, lat]",
            get_radius=90000, get_fill_color=[231, 76, 60, 45],
            get_line_color=[231, 76, 60, 170], stroked=True, filled=True,
            line_width_min_pixels=1, pickable=False))
    if getattr(wr, "source_lat", None) is not None:
        sdf = pd.DataFrame([{"lon": wr.source_lon, "lat": wr.source_lat,
                             "name": wr.source_name}])
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=sdf, get_position="[lon, lat]",
            get_radius=11000, get_fill_color=[17, 17, 17, 230], pickable=False))
        layers.append(pdk.Layer(
            "TextLayer", data=sdf, get_position="[lon, lat]", get_text="name",
            get_size=13, get_color=[17, 17, 17], get_pixel_offset=[0, -16]))
    if not adf.empty:
        layers.append(pdk.Layer(
            "IconLayer", data=adf, get_icon="icon_data",
            get_position="[lon, lat]", get_size=4, size_scale=10,
            get_angle="angle", get_color="color", pickable=True))

    view = pdk.ViewState(latitude=wr.lat, longitude=wr.lon, zoom=zoom, pitch=0)
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto",
                    map_style="light", height=560,
                    tooltip={"text": "{callsign}  |  {status}  |  NIC {nic}"})
