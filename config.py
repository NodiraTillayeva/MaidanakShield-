"""
Maidanak Sentinel - global configuration.

Everything that defines *where* and *what* the prototype watches lives here so the
dashboard, the figure exporter and the batch scripts all agree on one source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- regions
@dataclass(frozen=True)
class Region:
    """A geographic bounding box the system monitors."""
    key: str
    name: str
    lamin: float  # min latitude
    lomin: float  # min longitude
    lamax: float  # max latitude
    lomax: float  # max longitude

    @property
    def center(self) -> tuple[float, float]:
        return ((self.lamin + self.lamax) / 2, (self.lomin + self.lomax) / 2)

    def contains(self, lat: float, lon: float) -> bool:
        return self.lamin <= lat <= self.lamax and self.lomin <= lon <= self.lomax


# Primary theatre: the Central Asian corridor around the Maidanak Observatory.
# This is exactly the airspace EUROCONTROL's reporting framework does not cover.
CENTRAL_ASIA = Region(
    key="central_asia",
    name="Central Asia / Maidanak corridor",
    lamin=37.0,
    lomin=58.0,
    lamax=46.0,
    lomax=75.0,
)

REGIONS = {CENTRAL_ASIA.key: CENTRAL_ASIA}
DEFAULT_REGION = CENTRAL_ASIA


@dataclass(frozen=True)
class WatchRegion:
    """A circular area we poll for real GNSS interference, with a publicly
    documented suspected source used as an independent footprint reference."""
    key: str
    name: str
    lat: float
    lon: float
    dist_nm: float = 250.0          # adsb.lol max radius
    source_name: str = ""           # publicly documented interference source
    source_lat: float | None = None
    source_lon: float | None = None


# Real, currently-active interference theatres (publicly documented) plus the
# Central Asian corridor the product is pitched for. The demo scans these live.
WATCH_REGIONS = {
    "baltic": WatchRegion("baltic", "Baltic / Kaliningrad", 55.0, 21.0,
                          source_name="Kaliningrad (documented jamming)",
                          source_lat=54.71, source_lon=20.51),
    "gulf": WatchRegion("gulf", "Persian Gulf", 27.0, 52.0,
                        source_name="Strait of Hormuz region"),
    "black_sea": WatchRegion("black_sea", "Black Sea", 44.5, 35.0,
                             source_name="Crimea region",
                             source_lat=44.95, source_lon=34.10),
    "e_med": WatchRegion("e_med", "Eastern Mediterranean", 34.0, 34.5,
                         source_name="Eastern Mediterranean / Levant"),
    "central_asia": WatchRegion("central_asia", "Central Asia / Maidanak", 41.0, 66.9,
                                source_name="Maidanak corridor (deployment target)",
                                source_lat=38.673, source_lon=66.894),
}
DEFAULT_WATCH = "baltic"


# --------------------------------------------------------------------------- assets
# Maidanak Observatory - the real optical facility that gives the system its
# independent physical ground truth. Coordinates of the main site in the
# Pamir-Alay mountains of southern Uzbekistan (~2600 m altitude).
MAIDANAK_OBSERVATORY = {
    "name": "Maidanak Observatory",
    "lat": 38.673,
    "lon": 66.894,
    "elevation_m": 2593.0,
}

# Ground SDR sensor network (the "Maidanak Shield" layer). For the prototype we
# model a sparse civilian RTL-SDR mesh along the corridor's busiest airports.
SHIELD_SENSORS = [
    {"name": "Tashkent",   "lat": 41.257, "lon": 69.281},
    {"name": "Samarkand",  "lat": 39.700, "lon": 66.984},
    {"name": "Bukhara",    "lat": 39.775, "lon": 64.483},
    {"name": "Maidanak",   "lat": 38.673, "lon": 66.894},
    {"name": "Dushanbe",   "lat": 38.543, "lon": 68.825},
    {"name": "Almaty",     "lat": 43.352, "lon": 77.040},
]

# Regional airports used to synthesise realistic baseline traffic for the
# detection test-bench (great-circle routes between real fields).
REGIONAL_AIRPORTS = {
    "TAS": {"name": "Tashkent",   "lat": 41.257, "lon": 69.281},
    "SKD": {"name": "Samarkand",  "lat": 39.700, "lon": 66.984},
    "BHK": {"name": "Bukhara",    "lat": 39.775, "lon": 64.483},
    "DYU": {"name": "Dushanbe",   "lat": 38.543, "lon": 68.825},
    "ALA": {"name": "Almaty",     "lat": 43.352, "lon": 77.040},
    "FRU": {"name": "Bishkek",    "lat": 43.061, "lon": 74.477},
    "ASB": {"name": "Ashgabat",   "lat": 37.987, "lon": 58.361},
    "NQZ": {"name": "Astana",     "lat": 51.022, "lon": 71.467},
    "CIT": {"name": "Shymkent",   "lat": 42.364, "lon": 69.479},
}


# --------------------------------------------------------------------------- data sources
OPENSKY_STATES_URL = "https://opensky-network.org/api/states/all"
CELESTRAK_GPS_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle"
HTTP_TIMEOUT = 15  # seconds


# --------------------------------------------------------------------------- physics
@dataclass(frozen=True)
class PhysicsLimits:
    """Flight-envelope limits for a generic commercial jet, used by the Sentinel
    Twin to decide whether a reported state transition is physically possible."""
    max_speed_ms: float = 340.0          # ~ Mach 1.0 ground speed ceiling for airliners
    max_accel_ms2: float = 4.0           # comfortable longitudinal accel ceiling
    max_climb_ms: float = 30.0           # ~6000 ft/min
    max_turn_rate_deg_s: float = 6.0     # standard rate turn ~3 deg/s; 6 = aggressive
    max_teleport_km: float = 12.0        # plausible gap between 10 s samples at cruise
    min_sane_altitude_m: float = -300.0
    max_sane_altitude_m: float = 14000.0


PHYSICS = PhysicsLimits()


# --------------------------------------------------------------------------- detection
@dataclass(frozen=True)
class DetectionConfig:
    sample_dt_s: float = 10.0            # nominal ADS-B sampling cadence
    # weights for fusing the Sentinel Twin's anomaly cues into a 0..1 spoofing score.
    # The dead-reckoning residual (reported position vs the twin's own prediction) is
    # the primary cue: it catches slow pull-off, teleport, freeze and circling alike.
    weight_dr: float = 0.35             # dead-reckoning residual (twin vs reported)
    weight_speed: float = 0.20          # implied vs reported speed mismatch
    weight_bearing: float = 0.15        # implied vs reported heading mismatch
    weight_teleport: float = 0.15       # impossible position jump
    weight_turn: float = 0.08           # excessive turn rate
    weight_accel: float = 0.04          # excessive acceleration
    weight_altitude: float = 0.03       # altitude sanity / impossible climb
    weight_optical: float = 0.10        # observatory corroboration (fused at track level)
    dr_tolerance_km: float = 6.0        # residual that maps to a full anomaly (=1.0)
    # Operating point. With realistic manoeuvring traffic and covert attacks the
    # honest/attack score distributions overlap near ~0.28, so this threshold trades
    # a small miss rate (the most covert spoofs) for a very low false-alarm rate.
    alert_threshold: float = 0.28       # score above which an aircraft is flagged
    zone_grid_deg: float = 0.6          # interference-zone raster cell size
    zone_min_flags: int = 2             # flagged aircraft per cell to declare a zone


DETECT = DetectionConfig()


# --------------------------------------------------------------------------- alerting
@dataclass(frozen=True)
class AlertConfig:
    lookahead_min: float = 10.0         # how far ahead we project aircraft
    geofence_buffer_km: float = 50.0    # ring around a zone that triggers a pre-warn


ALERT = AlertConfig()
