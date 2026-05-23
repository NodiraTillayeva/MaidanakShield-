"""
Impact & business model - quantified with transparent, conservative assumptions.

The Round-2 brief asks for impact "for the industry, for the environment, for
society" and for a business model, and one of the two Round-2 webinars is "IT and
the environment". So we make the sustainability and economic case explicit and
*auditable*: every number below comes from a stated assumption you can challenge,
not a black box. All figures are illustrative order-of-magnitude estimates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ImpactAssumptions:
    # --- traffic exposed to the Central Asian corridor (illustrative) ---
    flights_per_day: int = 300            # daily flights transiting the corridor
    interference_days_frac: float = 0.20  # share of days with active interference
    flights_affected_frac: float = 0.30   # share of those flights actually affected

    # --- operational penalty of a GNSS disruption, per affected flight ---
    # Without pre-warning: reactive holding + indirect routing/diversion.
    penalty_min_reactive: float = 15.0
    # With Maidanak Sentinel: planned avoidance around a known zone.
    penalty_min_preempt: float = 6.0

    # --- fuel & emissions (A320-class, ICAO factors) ---
    cruise_burn_kg_per_min: float = 40.0  # ~2,400 kg/h
    co2_per_kg_fuel: float = 3.16         # ICAO standard jet-A factor
    fuel_usd_per_kg: float = 0.85         # ~ $0.85/kg jet fuel
    aircraft_op_usd_per_min: float = 110.0  # crew+lease+maintenance per airborne min

    # --- deployment cost (order-of-magnitude, illustrative) ---
    sentinel_node_usd: int = 200          # RTL-SDR + single-board computer + antenna
    sentinel_nodes: int = 20              # corridor sensor mesh
    gpspatron_node_usd: int = 6000        # commercial ground TDOA probe (ballpark)
    military_crpa_per_aircraft_usd: int = 40000  # CRPA antenna/aircraft (ballpark)


def environmental_economic(a: ImpactAssumptions = ImpactAssumptions()) -> dict:
    """Annual fuel, CO2 and cost avoided by pre-emptive mitigation vs reactive."""
    affected_per_day = a.flights_per_day * a.interference_days_frac * a.flights_affected_frac
    affected_per_year = affected_per_day * 365.0

    saved_min_per_flight = a.penalty_min_reactive - a.penalty_min_preempt
    saved_min_year = affected_per_year * saved_min_per_flight

    fuel_saved_kg = saved_min_year * a.cruise_burn_kg_per_min
    co2_saved_kg = fuel_saved_kg * a.co2_per_kg_fuel
    fuel_cost_saved = fuel_saved_kg * a.fuel_usd_per_kg
    delay_cost_saved = saved_min_year * a.aircraft_op_usd_per_min

    return {
        "affected_flights_per_year": round(affected_per_year),
        "minutes_saved_per_year": round(saved_min_year),
        "fuel_saved_tonnes_year": round(fuel_saved_kg / 1000.0, 1),
        "co2_saved_tonnes_year": round(co2_saved_kg / 1000.0, 1),
        "co2_saved_equiv_cars": round(co2_saved_kg / 4600.0),  # ~4.6 t CO2/car/yr
        "cost_saved_usd_year": round(fuel_cost_saved + delay_cost_saved),
    }


def affordability(a: ImpactAssumptions = ImpactAssumptions()) -> dict:
    """Per-corridor capital cost vs. the alternatives (order-of-magnitude)."""
    return {
        "Maidanak Shield (20-node mesh)": a.sentinel_node_usd * a.sentinel_nodes,
        "Commercial ground probes (1 site)": a.gpspatron_node_usd,
        "Military CRPA (1 aircraft)": a.military_crpa_per_aircraft_usd,
    }


def business_model() -> dict:
    """Revenue streams and target customers."""
    return {
        "streams": [
            ("Airspace-monitoring SaaS", "ANSPs · airports · regulators",
             "subscription per monitored FIR/corridor"),
            ("Sensor kits (Maidanak Shield)", "ANSPs · universities · partners",
             "low-cost hardware + install"),
            ("Sentinel Twin avionics licence", "airlines · OEMs · UAM/drone operators",
             "per-tail software licence"),
            ("Forensic & insurance reports", "insurers · investigators · militaries",
             "optically-verified incident reports"),
        ],
        "wedge": "Civil corridors with no coverage today (Central Asia, Caucasus, "
                 "Africa) - where military gear is unavailable and EUROCONTROL stops.",
    }


def use_cases() -> dict:
    """The brief asks for security, space AND defence - spell all three out."""
    return {
        "Security": "Protect civil aviation, airports and critical timing "
                    "infrastructure (power, finance, telecoms all rely on GNSS time) "
                    "from jamming/spoofing - for operators who will never buy military kit.",
        "Space": "Resilient PNT for the space-dependent economy: independent integrity "
                 "monitoring of GNSS, optical confirmation of satellite geometry, and a "
                 "ground-truth layer for future LEO-PNT and autonomous/UAM operations.",
        "Defence": "Wide-area, attributable interference mapping and pre-emptive "
                   "alerting in grey-zone airspace - dual-use, exportable, and "
                   "deployable where CRPA/M-code cannot go.",
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"environmental_economic": environmental_economic(),
                      "affordability": affordability(),
                      "assumptions": asdict(ImpactAssumptions())}, indent=2))
