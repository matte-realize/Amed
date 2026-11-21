"""FastAPI backend for Hospital Twin (Amed)."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from hospital_twin.schemas import SimulationComparison, UnitMap

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parent.parent
DEMO_DIR = REPO_ROOT / "data" / "demo"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

app = FastAPI(
    title="Hospital Twin API",
    description="Floor plan extraction, run schedule parsing, and nurse vs. courier-robot shift simulation.",
    version="0.1.0",
)


class SimulateRequest(BaseModel):
    unit_map: UnitMap
    run_description: str = Field(min_length=1)
    num_robots: int = Field(default=1, ge=1, le=3)
    shift_hours: float = Field(default=8.0, gt=0.0, le=24.0)


def _list_demo_plans() -> list[dict[str, str]]:
    if not DEMO_DIR.exists():
        return []
    plans: list[dict[str, str]] = []
    for path in sorted(DEMO_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            plan_id = path.stem
            label = plan_id.replace("_", " ").replace("-", " ").title()
            plans.append(
                {
                    "id": plan_id,
                    "name": label,
                    "filename": path.name,
                    "image_url": f"/api/plans/{plan_id}/image",
                }
            )
    return plans


def _resolve_demo_image(plan_id: str) -> Path:
    if not plan_id or "/" in plan_id or "\\" in plan_id or ".." in plan_id:
        raise HTTPException(status_code=400, detail="Invalid plan_id.")
    if not DEMO_DIR.exists():
        raise HTTPException(status_code=404, detail="Demo directory not found.")
    for ext in IMAGE_EXTENSIONS:
        candidate = DEMO_DIR / f"{plan_id}{ext}"
        if candidate.is_file():
            return candidate
    raise HTTPException(status_code=404, detail=f"Floor plan '{plan_id}' not found.")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/plans")
async def list_plans() -> dict[str, list[dict[str, str]]]:
    return {"plans": _list_demo_plans()}


@app.get("/api/plans/{plan_id}/image")
async def get_plan_image(plan_id: str) -> FileResponse:
    image_path = _resolve_demo_image(plan_id)
    return FileResponse(image_path)


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return bool(val)


@app.post("/api/extract")
async def extract_unit(request: Request) -> dict[str, Any]:
    from hospital_twin.extraction import (
        extract_unit_map,
        load_fallback_unit,
        verify_unit_map,
    )

    content_type = request.headers.get("content-type", "").lower()
    plan_id: Optional[str] = None
    use_fallback = False
    meters_per_pixel = 0.05
    uploaded_bytes: Optional[bytes] = None
    uploaded_suffix = ".png"

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object.")
        plan_id = body.get("plan_id")
        use_fallback = _parse_bool(body.get("use_fallback", False))
        if body.get("meters_per_pixel") is not None:
            meters_per_pixel = float(body["meters_per_pixel"])
    else:
        form = await request.form()
        raw_plan_id = form.get("plan_id")
        if isinstance(raw_plan_id, str) and raw_plan_id.strip():
            plan_id = raw_plan_id.strip()
        use_fallback = _parse_bool(form.get("use_fallback", False))
        raw_mpp = form.get("meters_per_pixel")
        if isinstance(raw_mpp, str) and raw_mpp.strip():
            meters_per_pixel = float(raw_mpp)
        upload = form.get("file")
        if upload is not None and hasattr(upload, "read"):
            uploaded_bytes = await upload.read()
            filename = getattr(upload, "filename", None) or "upload.png"
            suffix = Path(filename).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                uploaded_suffix = suffix

    if use_fallback:
        try:
            unit_map = load_fallback_unit(plan_id)
        except TypeError:
            unit_map = load_fallback_unit()
        stats = {
            "kept": len(unit_map.rooms) + len(unit_map.doors),
            "dropped": 0,
            "re_extracted": 0,
            "mean_confidence": 1.0,
            "source": "fallback",
        }
        return {
            "unit_map": unit_map.model_dump(),
            "verification": stats,
            "source": "fallback",
        }

    temp_path: Optional[str] = None
    try:
        if uploaded_bytes:
            with tempfile.NamedTemporaryFile(suffix=uploaded_suffix, delete=False) as tmp:
                tmp.write(uploaded_bytes)
                temp_path = tmp.name
            image_path = temp_path
        elif plan_id:
            image_path = str(_resolve_demo_image(plan_id))
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide a floor plan file upload, a valid plan_id, or set use_fallback=true.",
            )

        raw_unit = extract_unit_map(image_path, meters_per_pixel=meters_per_pixel)
        verified_unit, stats = verify_unit_map(image_path, raw_unit)
        return {
            "unit_map": verified_unit.model_dump(),
            "verification": stats,
            "source": "extracted",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@app.post("/api/simulate", response_model=SimulationComparison)
async def simulate(req: SimulateRequest) -> SimulationComparison:
    from hospital_twin.schedule_parser import parse_run_description
    from hospital_twin.simulator import simulate_shift

    available_rooms = [room.id for room in req.unit_map.rooms]
    if not available_rooms:
        raise HTTPException(status_code=400, detail="UnitMap contains no rooms.")

    try:
        schedule = parse_run_description(req.run_description, available_rooms)
        if not schedule:
            raise HTTPException(
                status_code=400,
                detail="Could not parse any scheduled runs from the description.",
            )
        comparison = simulate_shift(
            req.unit_map,
            schedule,
            shift_hours=req.shift_hours,
            num_robots=req.num_robots,
        )
        return comparison
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("hospital_twin.app:app", host="0.0.0.0", port=8000, reload=True)
