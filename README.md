# Maidanak Sentinel - technical code

Detection code for **Maidanak Sentinel**, a two-layer GNSS jamming/spoofing defence
that runs on real, open data. Airbus Fly Your Ideas 2026.

This folder contains **only the technical code and a data sample** - no slides, 3D
assets, or generated images.

## What it does

- **Maidanak Shield** ([src/detect/integrity.py](src/detect/integrity.py), [src/detect/rf_shield.py](src/detect/rf_shield.py)) - classifies real ADS-B navigation-integrity (NIC/NACp) and clusters degraded aircraft into live interference zones.
- **Sentinel Twin** ([src/detect/physics_twin.py](src/detect/physics_twin.py)) - an onboard kinematic digital twin that dead-reckons every fix to catch spoofing before the receiver is fooled.
- **Maidanak Observatory** ([src/detect/optical_verify.py](src/detect/optical_verify.py)) - uses the real GPS constellation (ephemeris) as ground truth a spoofer cannot move.
- **Pre-emptive alert** ([src/alert/preemptive.py](src/alert/preemptive.py)) - pre-warns aircraft on course into a flagged zone.

## Quickstart

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# live detection over the worst current theatre (real data)
./.venv/bin/python -c "from src.pipeline import scan_live; print(scan_live().to_string(index=False))"

# full live result for one theatre
./.venv/bin/python -m src.pipeline            # optional labelled-attack benchmark
./.venv/bin/python -c "from src.pipeline import run_live_detection as r; x=r('baltic'); print(x.summary)"

# interactive dashboard
./.venv/bin/streamlit run app.py
```

To run fully offline against the bundled sample instead of live feeds, point the
ingest at `data_sample/` (e.g. `pandas.read_csv('data_sample/adsb_baltic_live_sample.csv')`)
or pass `use_cache=True` after copying a sample into `data/`.

## Layout

```
.
├── app.py                 # Streamlit operator console
├── export_figures.py      # generates figures from real data
├── config.py              # regions, watch theatres, detector tuning
├── requirements.txt
├── src/
│   ├── geo.py             # great-circle geodesy
│   ├── impact.py          # environmental / economic / business model
│   ├── planemap.py        # live map (airplane icons) for the dashboard
│   ├── pipeline.py        # run_live_detection (real) + run_testbench (sim)
│   ├── ingest/            # adsb.py (real NIC), gnss.py (GPS TLE), opensky.py
│   ├── detect/            # integrity, rf_shield, physics_twin, optical_verify
│   ├── simulate/          # scenario + attacks (labelled test-bench, optional)
│   ├── alert/             # preemptive ground->air handoff
│   └── eval/              # metrics (ROC / precision / recall)
└── data_sample/           # real ADS-B + GPS TLE samples + labelled test-bench
```

## What is real vs simulated
- **Real:** aircraft + GNSS integrity (adsb.lol / airplanes.live), GPS constellation
  (Celestrak). All live detection runs on these.
- **Simulated (labelled):** the attack test-bench in `src/simulate/`, used only to
  measure detection accuracy with ground-truth labels. Not used by the live demo.

## Data sources
adsb.lol · airplanes.live · OpenSky Network · Celestrak · Maidanak Observatory.
See [data_sample/README.md](data_sample/README.md).
