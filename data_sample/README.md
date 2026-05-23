# Data sample

Representative samples of the data the system consumes. The detector runs on the
**real** ones; the labelled test-bench is used only to *measure* accuracy.

| File | What it is | Source / use |
|---|---|---|
| `adsb_baltic_live_sample.csv` | A live snapshot of real aircraft over the Baltic / Kaliningrad, with the ADS-B navigation-integrity fields **NIC, NACp, Rc** and our derived `status` (healthy / degraded / lost). | **Real** (adsb.lol / airplanes.live). The core input the Maidanak Shield layer detects interference from. |
| `gps_constellation_tle.txt` | Real GPS operational two-line elements (the live constellation). | **Real** (Celestrak). Used to compute the true sky geometry - the observatory / optical ground-truth layer. |
| `testbench_labeled_tracks_sample.csv` | Synthetic cruise traffic with injected spoofing/jamming, fully time-sampled. Columns include `is_attack` and `attack_type`. | **Simulated, labelled.** Used only to measure detection ROC / precision / recall (real jamming is rare and unlabelled). |
| `testbench_truth_labels_sample.csv` | Per-aircraft ground-truth labels for the test-bench (attack type + onset). | Labels for the file above. |

## Key columns

**`adsb_baltic_live_sample.csv`** - `track_id` (ICAO24), `callsign`, `lat`, `lon`,
`alt_m`, `velocity` (m/s), `heading`, `nic`, `nac_p`, `rc_m`, `airborne`,
`integrity_score` (0 healthy .. 1 lost), `status`.

**`testbench_labeled_tracks_sample.csv`** - `track_id`, `t` (s), `lat`, `lon`,
`alt`, `velocity`, `heading`, `vertical_rate`, `is_attack`, `attack_type`.

> Live feeds change minute to minute, so re-running the code will fetch a fresh
> snapshot; this CSV is a frozen example for reproducibility.
