# Amed — 3D Hospital Unit Twin (Gemini 4 NYC Hackathon)

Built completely from scratch during the **Gemini 4 NYC Hackathon (Nov 21, 2026)** for **Element 119**.

> "Nurses spend part of every shift walking supplies around the unit. Give us a floor plan, and we show what changes when robots take those trips."

## Model Jobs & Pipeline
Three Gemini 4 models carry distinct jobs (Carbon omitted to stay within the 4-hour build window):
1. **Gemini 4 Neon (`gemini-4-neon`)**: Extracts rooms (`label`, `type`, `polygon`, `obstacles`, `autonomy_tier`, `autonomy_reason`) and doors (`position`, `rooms_connected`, `width_m`, `robot_passable`) from a floor plan image into `unit.json`.
2. **Decisions API (`gemini-decision-1`)**: Verifies each extracted room and door crop in parallel (`ThreadPoolExecutor`) with calibrated confidence; re-extracts or drops low-confidence items into `verified_unit.json`.
3. **Gemini 4 Hydrogen (`gemini-4-hydrogen`)**: Parses spoken or typed shift run descriptions (e.g., *"pharmacy pickup every 30 minutes, specimens hourly, supply restock twice a shift"*) into structured `ScheduledRun` items.
4. **NetworkX + SimPy**: Converts `verified_unit.json` into a weighted meter route graph and simulates nurse-only (`1.2 m/s`) vs. 1–3 courier robots (`0.8 m/s`) over an 8-hour shift.
5. **Interactive 3D Twin (Three.js + 2D fallback)**: Renders extruded 3D rooms, interior fixtures (`bed`, `desk`, `med_cart`, `shelf`), doorway arches, **Autonomy Zone** overlays (`Autonomous` / `Assisted` / `Human-Only`), and animated 3D nurse & robot figures side-by-side.

## Proof: Live Extraction Evaluation (10 Unseen CubiCasa5K Plans)
Evaluated live against 10 real, unseen architectural floor plans from `CubiCasa5K` (`Tristan12138/cubicasa5k-skv4` test split, `data/eval/cubicasa_10/`):

| Metric | Baseline (Neon only) | Verified (Neon + Decisions API) | Delta |
| --- | ---: | ---: | ---: |
| Room Detection Rate | 100.0% | 83.5% | -16.6 pp |
| Door Detection Rate | 85.9% | 56.2% | -29.8 pp |
| Door Precision | 95.0% | 100.0% | +5.0 pp |
| Mean Room IoU | 0.797 | 0.793 | -0.004 |

### What Failed & What We Changed
- **Coordinate Frame Mismatch (fixed in `2d7119c`):** Initial eval scored 17.9% room detection because `cubicasa5k-skv4` ground-truth polygons are in a `1024.0` reference frame while the PNGs are ~`256×335` px. Scaling by `max(w, h) / frame` revealed Neon's true `100.0%` room recall and `0.797` mean IoU.
- **Over-Pruning Unlabeled Rooms (fixed in `bcc0185`):** Decisions API initially dropped valid rooms (75.6% → 83.5% after fix) when asked to verify hospital-style labels against residential/unlabeled plans. Switching the check to structural room validity recovered +7.9 pp room recall while preserving `100.0%` door precision.
- **Remaining Breaking Points:** On low-res (`~256×335`) thumbnails, small door crops lose wall-arc context so Decisions API conservatively rejects ~30% of true doors; handwritten or unscaled plans without a reference width require manual `meters_per_pixel` calibration.

## Stated Simulation Assumptions
- Nurse walking speed: `1.2 m/s` | Courier robot speed: `0.8 m/s`
- Default scale: `0.05 m/pixel` (configurable per plan)
- Scope: Logistics trips only (`pharmacy_pickup`, `lab_specimen_drop`, `supply_restock`); no clinical or patient-care tasks. Time-saved figures are scenario estimates from these stated assumptions.

## Quickstart
```bash
cp .env.example .env   # add GEMINI_API_KEY
uv sync
PYTHONPATH=src .venv/bin/pytest tests -v
PYTHONPATH=src .venv/bin/uvicorn hospital_twin.app:app --reload --port 8000
```
Open `http://127.0.0.1:8000`.
