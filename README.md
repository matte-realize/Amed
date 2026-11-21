# Amed — Hospital Logistics World Model

Amed converts any hospital unit floor plan and facility specification into an interactive 3D digital twin. Operations teams and robotics engineers use it to test existing and custom courier platforms, catch doorway and layout blockers early, and stress-test nurse vs. robot workflows before deploying hardware.

> *"Nurses spend part of every shift walking supplies around the unit. Give us a floor plan, and we show what changes when robots take those trips."*

## Core Capabilities

1. **Floor Plan & Facility Spec Extraction (`gemini-4-neon` + `gemini-decision-1`)**
   - Extracts rooms (`label`, `type`, `polygon`, `obstacles`, `autonomy_tier`) and doorways (`position`, `rooms_connected`, `width_m`, `robot_passable`) from floor plan images and optional `.pdf` / `.txt` / `.csv` facility specs.
   - Verifies extracted room and door crops in parallel via the Decisions API (`gemini-decision-1`), re-extracting or pruning low-confidence detections into `verified_unit.json`.
2. **Natural-Language & Voice Run Schedules (`gemini-4-hydrogen`)**
   - Parses spoken (Web Speech API) or typed logistics routines (e.g., *"pharmacy pickup every 30 minutes, specimens hourly, supply restock twice a shift"*) into structured `ScheduledRun` schedules.
3. **Existing & Custom Robotics Simulation (`NetworkX` + `SimPy`)**
   - Compares **Nurse-Only** (`1.2 m/s`) vs. **1–3 Courier Robots** across 5 platforms:
     - **Existing:** `Aethon TUG + 3-Bay Pod` (`0.76 m/s`, `0.90m` clearance), `Diligent Moxi Cobot` (`0.90 m/s`, `0.85m`), `Pneumatic Tube Hybrid` (`3.50 m/s`, specimens/meds).
     - **Custom Amed Concepts:** `Amed Pre-Stage Cart` (`1.10 m/s`, 4-bay emergency staging), `Amed Slim Rover` (`0.85 m/s`, `0.72m` narrow-door clearance).
   - Models 20s badge-and-unlock handoffs, 3-compartment pod batching, doorway clearance fit checks, live **Custody Log** tracking, and `>10 min` uncollected delivery flagging.
4. **World Model Stress Sweeps & Calibration**
   - Stress-test modes: `routine`, `chaotic`, `surge`, `robot_failure`, `pharmacy_delay`, `staffing_gap`, `crash_cart_missing`, plus `EMS Pre-Stage` advance signal modeling.
   - Multi-seed fleet sweeps (`POST /api/stress-test`) locate tipping points, rank bottlenecks, generate an **Autonomy Plan** (`use_robot`, `redesign_process`, `keep_human`), and export Markdown After-Action Reports.
   - Mock-drill calibration (`POST /api/calibrate`), live staff room/door editing (`POST /api/review-unit`), NEMSIS-style EMS arrival adapter (`POST /api/ems-feed`), and regional coalition surge balancing (`POST /api/coalition-surge`).
5. **Interactive 3D Twin (Three.js + 2D Canvas Fallback)**
   - Renders extruded 3D rooms, clinical equipment (hospital beds with IV poles, curved nurse stations, Pyxis-style med cabinets, supply racks, lab stations), red/green doorway fit checks, distinct 3D robot meshes, and articulated 3D nurses side-by-side.

## Validation & Benchmarks

### Unseen Floor Plan Extraction (`data/eval/cubicasa_10/`)
Evaluated against 10 unseen architectural floor plans from `CubiCasa5K` (`Tristan12138/cubicasa5k-skv4` test split):

| Metric | Baseline (Neon only) | Verified (Neon + Decisions API) | Delta |
| --- | ---: | ---: | ---: |
| Room Detection Rate | 100.0% | 83.5% | -16.6 pp |
| Door Detection Rate | 85.9% | 56.2% | -29.8 pp |
| Door Precision | 95.0% | 100.0% | +5.0 pp |
| Mean Room IoU | 0.797 | 0.793 | -0.004 |

- **Key Fixes:** Resolved `1024.0` reference-frame scaling mismatch (`2d7119c`) and switched Decisions API room checks to structural validity (`bcc0185`), preserving `100.0%` door precision.
- **Known Limits:** Low-resolution (`~256×335`) thumbnails lose wall-arc context on ~30% of true doors; unscaled plans require manual `meters_per_pixel` calibration.

### Degradation Stress Test (`data/eval/stress_results.json`)

| Variant | Effective Res | Baseline Rooms | Verified Rooms | Verified Label Acc | Verified Doors | Status |
| --- | :---: | ---: | ---: | ---: | ---: | --- |
| 1.0x (Clean) | `800x500` | 100.0% | 100.0% | 100.0% | 100.0% | **PASS (>=75%)** |
| 0.5x Downscale | `400x250` | 100.0% | 100.0% | 100.0% | 100.0% | **PASS (>=75%)** |
| 0.25x Downscale | `200x125` | 100.0% | 100.0% | 100.0% | 100.0% | **PASS (>=75%)** |
| Low Contrast + Blur | `800x500` | 100.0% | 100.0% | 100.0% | 100.0% | **PASS (>=75%)** |
| Rotated 15° | `800x500` | 100.0% | 100.0% | 100.0% | 20.0% | **BREAKING POINT (<75%)** |

### Sourced Physical Inputs & Scope

| Input | Source | Value |
| --- | --- | --- |
| Unit Floor Plan | VA Medical/Surgical Inpatient Units & ICU Design Guide (Nov 2011) | Standard med/surg unit (`0.05 m/px` default) |
| Unit Size Check | VA Space Planning Criteria, Chapter 100 | 17–33 beds per inpatient unit |
| Courier Robot | Aethon TUG specifications | `76 cm/s` (`0.76 m/s`), `96.5 cm` long |
| Demand Sanity Check | Aethon, via RIA | ~10,000 med orders/week at a 200-bed hospital |

- **Scope:** Logistics trips only (`pharmacy_pickup`, `lab_specimen_drop`, `supply_restock`); no clinical or patient-care tasks. Time-saved figures are scenario estimates from stated assumptions.

## Quickstart

```bash
cp .env.example .env   # add GEMINI_API_KEY
uv sync
.venv/bin/pytest -v
PYTHONPATH=src .venv/bin/uvicorn hospital_twin.app:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/extract` | Extract and verify a `UnitMap` from a floor plan image and optional facility spec |
| `POST` | `/api/simulate` | Parse a run schedule and compare nurse-only vs. courier-robot shift metrics |
| `POST` | `/api/stress-test` | Run multi-seed fleet stress sweeps to locate tipping points and build an autonomy plan |
| `POST` | `/api/calibrate` | Calibrate simulation assumptions against observed mock-drill timings |
| `POST` | `/api/review-unit` | Apply staff room and doorway overrides to an extracted `UnitMap` |
| `POST` | `/api/ems-feed` | Generate an EMS arrival pre-staging dispatch plan from an incoming signal |
| `POST` | `/api/coalition-surge` | Evaluate regional coalition surge balancing across units |
| `GET` | `/api/robots` | List supported existing and custom courier robot profiles |
| `GET` | `/api/struggles` | List modeled hospital logistics pain points and operational blockers |

*Originally created Nov 21, 2026.*
