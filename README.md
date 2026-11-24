# V2X-TwinStream

**A synchronized cyber-physical telemetry and branching incident-response video-track system for EV, CAV and AV security.**

TwinStream aligns three feeds on one deterministic millisecond master clock: the *physical* view (what actually happens to cars and chargers), the *cyber* telemetry (BSM, OCPP, CAN, LiDAR) and a *defender walkthrough*. A decision gate branches the scenario, and every branch re-computes the physical consequence. A consequence-aware audit then scores intrusion detectors by what they let happen (minimum time-to-collision, voltage and frequency deviation, overcurrent exposure) and by detection latency in milliseconds, not only by accuracy.

* Clean-room, MIT licensed, no proprietary courseware, no network access needed at run time except the optional dataset downloads.
* **No random numbers anywhere.** All attacks are injected into *recorded real data* and labelled; all consequence models are deterministic functions.
* Python 3.12+, FastAPI, vanilla-JS front end (no build step).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                             # 49 tests, ~10 s, use only the bundled samples
uvicorn backend.main:app --port 8000                  # open http://127.0.0.1:8000
```

Tracks 1, 2, 4 and 5 run from the bundled samples (public-domain NGSIM rows, CC-BY Norway sessions). Track 3 and the AV benchmarks need KITTI, which is CC BY-NC-SA and therefore **not redistributed**:

```bash
python scripts/fetch_data.py ngsim ev kitti           # full raw data (~120 MB NGSIM+EV, KITTI drive 0001)
python benchmarks/run_sync_and_ids_eval.py --rerun    # recompute every number (~10 min)
python benchmarks/generate_paper_figures.py           # results/figures/*.{svg,pdf,jpg}
```

## The five tracks

| Track | Domain | Real data | Attack | Gate |
|---|---|---|---|---|
| 1 | CAV | NGSIM Peachtree trajectories as J2735 BSMs | +12 m forged leader position | ignore / fall back to ranging / wait |
| 2 | EV | Norway EV sessions (OCPP-J frames, IEC 61851 pilot) | `SetChargingProfile` to 32 A with falsified `MeterValues` | ignore / re-issue rated profile / wait |
| 3 | AV | KITTI raw drive (OXTS-derived CAN, Velodyne) | rogue −5 m/s² brake request on CAN | ignore / cancel override / wait |
| 4 | CAV | NGSIM | stealthy ramp drift; *you choose the detector before the attack* | IF / RF / RF+physics / physics |
| 5 | CTF | Norway session + generated footage | forged OCPP message in a PCAP | verify provenance first, or not |

Front end: 3 synchronized feeds (physical POV canvas or video, live telemetry, defender walkthrough), a scrubber, and a decision-gate modal. Telemetry arrives frame-aligned from the server, so what is drawn at video frame *k* equals `GET /api/tracks/{id}/frame/{k}`.

## Synchronization

* Video frame *k* at *fps* maps to an **integer** millisecond `t0 + (2k·1000 + fps)//(2·fps)` (round-half-up, no floating-point drift; at most 0.5 ms from the ideal instant, 0 ms for 25/50 fps).
* Each stream has a clock offset and skew; a cross-correlation estimator recovers an unknown offset (max error on real BSM speed shifted by known offsets: 0.0007 ms).
* Determinism: the same track and decisions yield bit-identical telemetry (SHA-256 digest, `/api/tracks/{id}/digest`). Telemetry **before** the gate is identical across all branches (tested).
* Measured on all branches: interpolation query error 0 ms; nearest-sample error ≤ half a sample period; hold error ≤ one period (these are definitions, reported in `results/sync_eval.json`). Real KITTI Velodyne timestamps lag OXTS by 11–24 ms (mean 17 ms); the engine keeps the real timestamps unless you ask for the nominal grid (the AV track uses a nominal 100 ms grid because the consequence models step at 0.1 s).
* Some branches share a post-gate digest when the options have the same physical effect (e.g. Track 3, "wait 1 s" lets the 1.5 s override expire the same way as "ignore"; Track 5 telemetry does not depend on the choice). That is expected, not a bug.

## Consequence-aware IDS audit (honest summary)

Detectors: **Isolation Forest** (trained on benign only), **Random Forest** (supervised), and the **physics verifier** (`v_{t+1} = v_t + aΔt`, `Δx = vΔt`, V2X-vs-ranging gap for BSMs; `P = V·I·cosφ`, feeder power balance and CC-CV monotonicity for OCPP; command-vs-response and free-space checks for AV). ML thresholds are calibrated to 1 % FPR on held-out groups; physics thresholds are calibrated on real benign data. CAV uses 9-fold group-held-out CV over 27 real NGSIM leader-follower pairs (243 test cases).

**CAV (BSM false-data injection)** — benign false-alarm rate: physics 0.00, IF 0.33, RF 0.56.

| Attack | No defence collisions | IF det. / latency / collisions | RF det. / latency / collisions | Physics det. / latency / collisions |
|---|---|---|---|---|
| step offset +12 m | 0.85 | 0.13 / 9.8 s / 0.83 | 0.96 / 1.85 s / 0.24 | 1.00 / 0.2 s / 0.00 |
| ramp drift | 0.26 | 0.09 / 12.8 s / 0.26 | 0.67 / 5.2 s / 0.13 | 1.00 / **3.2 s** / 0.00 |
| speed spoof +4 m/s | 0.96 | 0.46 / 7.7 s / 0.83 | 0.96 / 1.7 s / 0.30 | 1.00 / 0.2 s / 0.00 |
| stale replay 3 s | 0.00 | 0.22 / 11 s / 0.00 | 0.89 / 1.85 s / 0.00 | 0.98 / 0.2 s / 0.00 |

An ablation, **RF + physics-residual features**, matches the physics verifier (benign FA 0.07, ramp-drift latency 0.3 s), so the gain comes from the physics *features*, not from the hand-written rule alone. Ramp drift is the physics verifier's slowest case (3.2 s).

**EV (OCPP)** — 6 real busy 30-min windows, leave-window-out ML. Physics benign FA 0.00; stealth meter FDI (reported values hide the overload): physics detects 6/6 in 2 s, RF only 1/6 (and late), IF 1/6; an RF given the feeder residual as a feature also detects it (benign FA 0.33). **Where physics loses:** for the oscillation attack, RF reacts in 2 s versus 12 s for physics, and RF limits the frequency excursion to ~0.01 Hz versus 0.93 Hz. Note the supervised RF has seen that attack class in training; the physics verifier has not.

**AV** — CAN brake tamper: physics and RF both alert in 200 ms (min TTC 8.3 s versus 3.3 s without defence); IF misses it. LiDAR ghost objects: free-space check AUC 0.97 (dense) / 0.90 (sparse) at 11 % false alarms, versus 0.75 / 0.59 for IF and 0.83 / 0.69 for RF at ~50 % false alarms. This check is weak: real objects and ghosts overlap in the statistic, and a sparse ghost is detected only 68 % of the time.

## Limitations (please read)

* **Attacks are injected, consequences are simulated.** There is no real attack capture. The follower (CACC), feeder/island and following-car models are simple and their constants are declared in `backend/domains.py`; conclusions about *ordering* of detectors are more trustworthy than absolute TTC or Hz values.
* **EV power curves are reconstructed.** The Norway dataset has only plug-in/out times and delivered kWh; a CC-CV profile conserving the real energy is built from it (flagged `reconstructed_from_real_session_energy`). Charger rating is inferred from the data. ACN-Data requires a token, so only an adapter (`load_acn_json`) is included; VeReMi/BSMAttacker are not used.
* NGSIM clips acceleration at ±3.74 m/s²; physics thresholds were recalibrated on real benign data accordingly. AV CAN signals are derived from OXTS (not a real CAN log) and the AV evaluation uses a single 10.8 s KITTI drive, so the AV numbers are a demonstration, not a benchmark.
* The EV evaluation has few windows (6) and one location; confidence intervals are wide.
* Track 4's case is a representative illustration chosen by a fixed rule (see `scenarios/README.md`), not an average.
* The stego strip is deliberately visible; it is a CTF puzzle, not a covert channel. The PCAP uses zero-valued TCP options and deterministic WebSocket masking keys for reproducibility.

## API

`GET /api/tracks` · `/api/tracks/{id}` · `/path?decisions=` · `/timeline?decisions=&k0=&k1=&step=` · `/frame/{k}` · `/scrub?t_ms=` · `/digest` · `/alignment` · `/api/tracks/track3_av_can/lidar/{k}` · `/api/audit` · `/api/ctf/{video,frame.png,manifest,artifacts/{name}}` · `POST /api/ctf/{extract,verify,submit}`. Interactive docs at `/docs`.

CTF tooling: `python -m backend.stego_manifest extract data/ctf/incident.mp4` decodes the hash; `python -m backend.stego_manifest pcap capture.pcap` lists the OCPP messages with source addresses.

## Layout

```
backend/   main.py (API) · tracks.py (runtime) · sync_engine.py · consequence_auditor.py · domains.py · stego_manifest.py · parsers/{j2735_bsm,ocpp_evse,av_sensor}_parser.py
scenarios/ track1..track5.json         frontend/ index.html app.js styles.css
benchmarks/ run_sync_and_ids_eval.py · generate_paper_figures.py        tests/ test_sync_engine.py · test_consequence_auditor.py
scripts/   fetch_data.py · make_samples.py · make_scenarios.py          data/samples/ (public-domain / CC-BY subsets)
```

## Data sources and licences

* NGSIM Peachtree Street, US DOT (public domain), via the data.transportation.gov API.
* *Electric vehicle charging dataset with 35,000 charging sessions from 12 residential locations in Norway*, Zenodo 13896176, **CC BY 4.0** (a 9,757-session subset for one location is bundled).
* KITTI raw `2011_09_26_drive_0001`, Geiger et al., **CC BY-NC-SA 3.0** (downloaded at run time, never redistributed here; do not use Track 3 outputs commercially).
* Standards referenced: SAE J2735 (BSM Part I), OCPP 1.6J / 2.0.1, ISO 15118, IEC 61851-1.

## Citation

> Author(s). *V2X-TwinStream: A Synchronized Cyber-Physical Telemetry and Branching Incident-Response Video Track System for EV, CAV and AV Security.* Software, 2026. (Author placeholder: fill in before publishing.)
