"""Floor plan extraction (Gemini 4 Neon) and verification (Decisions API)."""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import threading
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

from hospital_twin.schemas import Door, Point, Room, RoomType, UnitMap

NEON_MODEL = "gemini-4-neon"
DECISION_MODEL = "gemini-decision-1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = _REPO_ROOT / "data" / "demo"
FALLBACK_UNIT_PATH = DEMO_DIR / "fallback_unit.json"
ICU_WING_FALLBACK_PATH = DEMO_DIR / "icu_wing_fallback.json"
ST_ELIZABETHS_WARD_FALLBACK_PATH = DEMO_DIR / "st_elizabeths_ward_fallback.json"


class _ExtractedRoom(BaseModel):
    id: str
    label: str
    type: RoomType
    polygon: list[Point]
    centroid: Optional[Point] = None
    autonomy_tier: Literal["autonomous", "assisted", "human_only"] = "assisted"
    autonomy_reason: str = ""
    proposed_solution: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    obstacles: list[str] = Field(default_factory=list)


class _ExtractedDoor(BaseModel):
    id: str
    position: Point
    rooms_connected: list[str] = Field(min_length=2, max_length=2)
    width_m: float = 1.0
    robot_passable: bool = True


class _ExtractedUnitSchema(BaseModel):
    rooms: list[_ExtractedRoom]
    doors: list[_ExtractedDoor]


def _get_client(api_version: Optional[str] = None) -> genai.Client:
    """Create a Google GenAI client, loading .env if GEMINI_API_KEY is not set."""
    if not os.environ.get("GEMINI_API_KEY"):
        load_dotenv(_REPO_ROOT / ".env")
    if api_version:
        return genai.Client(http_options={"api_version": api_version})
    return genai.Client()


def compute_centroid(polygon: list[Point]) -> Point:
    """Compute the 2D centroid of a polygon, falling back to vertex mean."""
    if not polygon:
        return Point(x=0.0, y=0.0)
    if len(polygon) < 3:
        return Point(
            x=sum(p.x for p in polygon) / len(polygon),
            y=sum(p.y for p in polygon) / len(polygon),
        )

    area_acc = 0.0
    cx_acc = 0.0
    cy_acc = 0.0
    n = len(polygon)
    for i in range(n):
        p0 = polygon[i]
        p1 = polygon[(i + 1) % n]
        cross = p0.x * p1.y - p1.x * p0.y
        area_acc += cross
        cx_acc += (p0.x + p1.x) * cross
        cy_acc += (p0.y + p1.y) * cross

    if abs(area_acc) < 1e-6:
        return Point(
            x=sum(p.x for p in polygon) / n,
            y=sum(p.y for p in polygon) / n,
        )

    factor = 1.0 / (3.0 * area_acc)
    return Point(x=cx_acc * factor, y=cy_acc * factor)


def _image_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _image_to_bytes_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def crop_room_region(
    image: Image.Image,
    room: Room,
    padding: int = 16,
) -> Image.Image:
    """Crop the bounding box of a room polygon (with padding) from the floor plan."""
    width, height = image.size
    if room.polygon:
        xs = [p.x for p in room.polygon]
        ys = [p.y for p in room.polygon]
        x0 = max(0, int(min(xs)) - padding)
        y0 = max(0, int(min(ys)) - padding)
        x1 = min(width, int(max(xs)) + padding)
        y1 = min(height, int(max(ys)) + padding)
    elif room.centroid is not None:
        cx, cy = int(room.centroid.x), int(room.centroid.y)
        box_r = max(32, padding * 2)
        x0 = max(0, cx - box_r)
        y0 = max(0, cy - box_r)
        x1 = min(width, cx + box_r)
        y1 = min(height, cy + box_r)
    else:
        return image.copy()

    if x1 <= x0 or y1 <= y0:
        return image.copy()
    return image.crop((x0, y0, x1, y1))


def crop_door_region(
    image: Image.Image,
    door: Door,
    radius: int = 40,
) -> Image.Image:
    """Crop a square region centered on a door's pixel position."""
    width, height = image.size
    cx = int(round(door.position.x))
    cy = int(round(door.position.y))
    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(width, cx + radius)
    y1 = min(height, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return image.copy()
    return image.crop((x0, y0, x1, y1))


def extract_unit_map(
    image_path: str,
    meters_per_pixel: float = 0.05,
    spec_path: Optional[str] = None,
) -> UnitMap:
    """Extract rooms and doors from a floor plan image (+ optional spec doc) using Gemini 4 Neon."""
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        width, height = rgb_img.size
        img_bytes = _image_to_bytes_png(rgb_img)

    spec_parts: list[Any] = []
    if spec_path:
        sp = Path(spec_path)
        if sp.is_file():
            ext = sp.suffix.lower()
            if ext == ".pdf":
                spec_parts.append(types.Part.from_bytes(data=sp.read_bytes(), mime_type="application/pdf"))
            else:
                spec_text = sp.read_text(encoding="utf-8", errors="replace")[:12000]
                spec_parts.append(f"SUPPORTING FACILITY SPEC DOCUMENT ({sp.name}):\n{spec_text}")

    prompt = (
        f"Analyze this architectural floor plan image ({width}x{height} pixels).\n"
        "Extract every distinct room/corridor and every door connecting two rooms/corridors.\n"
        "For each room provide:\n"
        "- id: short snake_case identifier (e.g. 'room_101', 'nurse_station', 'corridor_main')\n"
        "- label: exact human-readable name printed on the plan (or short descriptive name if unlabeled)\n"
        "- type: one of patient_room, nurse_station, medication_room, supply_room, lab_drop, corridor, other\n"
        "- polygon: list of {x, y} pixel coordinates tracing the room boundary\n"
        "- centroid: {x, y} pixel center of the room\n"
        "- autonomy_tier: be conservative—only mark open corridors/lobbies as 'autonomous'; mark medication/supply/lab/nurse_station as 'assisted' (requires door actuation or staff handoff); mark patient/isolation rooms as 'human_only'\n"
        "- autonomy_reason: brief 1-sentence reason why this zone is currently autonomous, assisted, or human-only\n"
        "- proposed_solution: concrete autonomous workflow/hardware solution for this zone (e.g. 'Badge-actuated door + exterior courier drop bin' or 'AMR corridor transit lane')\n"
        "- prerequisites: 2-3 specific physical/workflow prerequisites required before full autonomy (e.g. ['Auto-door opener', '1.2m clear turning radius'])\n"
        "- obstacles: visible interior fixtures (e.g. ['bed', 'desk', 'med_cart', 'shelf'])\n"
        "For each door provide:\n"
        "- id: short snake_case identifier (e.g. 'door_101')\n"
        "- position: {x, y} pixel coordinate of the doorway\n"
        "- rooms_connected: exactly 2 room ids that this door connects\n"
        "- width_m: estimated doorway clear width in meters\n"
        "- robot_passable: true if width_m >= 0.90m and unobstructed, else false.\n"
    )

    client = _get_client()
    contents: list[Any] = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        *spec_parts,
        prompt,
    ]
    response = client.models.generate_content(
        model=NEON_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_ExtractedUnitSchema,
            thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
        ),
    )

    if getattr(response, "parsed", None) is not None:
        raw = response.parsed
        if isinstance(raw, BaseModel):
            data = raw.model_dump()
        else:
            data = dict(raw)
    else:
        data = json.loads(response.text)

    rooms: list[Room] = []
    for item in data.get("rooms", []):
        poly = [Point.model_validate(pt) for pt in item.get("polygon", [])]
        centroid_raw = item.get("centroid")
        centroid = Point.model_validate(centroid_raw) if centroid_raw else compute_centroid(poly)
        rooms.append(
            Room(
                id=item["id"],
                label=item["label"],
                type=item["type"],
                polygon=poly,
                centroid=centroid,
                autonomy_tier=item.get("autonomy_tier", "assisted"),
                autonomy_reason=item.get("autonomy_reason", ""),
                proposed_solution=item.get("proposed_solution", ""),
                prerequisites=list(item.get("prerequisites", [])),
                obstacles=list(item.get("obstacles", [])),
            )
        )

    doors = [
        Door(
            id=d["id"],
            position=Point.model_validate(d["position"]),
            rooms_connected=list(d["rooms_connected"]),
            width_m=float(d.get("width_m", 1.0)),
            robot_passable=bool(d.get("robot_passable", True)),
        )
        for d in data.get("doors", [])
    ]

    return UnitMap(
        image_width=width,
        image_height=height,
        meters_per_pixel=meters_per_pixel,
        rooms=rooms,
        doors=doors,
    )


def _re_extract_room_label(
    crop: Image.Image,
    room: Room,
    neon_client: genai.Client,
) -> Optional[Room]:
    """Attempt targeted re-extraction of a low-confidence room crop using Gemini 4 Neon."""
    class _RoomPatch(BaseModel):
        label: str
        type: RoomType

    try:
        response = neon_client.models.generate_content(
            model=NEON_MODEL,
            contents=[
                types.Part.from_bytes(data=_image_to_bytes_png(crop), mime_type="image/png"),
                (
                    f"This cropped hospital floor plan region was tentatively labeled '{room.label}' "
                    f"(type '{room.type}'). Confirm or correct the exact room label and type."
                ),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_RoomPatch,
                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            patch = parsed.model_dump() if isinstance(parsed, BaseModel) else dict(parsed)
        else:
            patch = json.loads(response.text)
        return room.model_copy(update={"label": patch["label"], "type": patch["type"]})
    except Exception:
        return None


def verify_unit_map(
    image_path: str,
    unit_map: UnitMap,
    confidence_threshold: float = 0.55,
    re_extract_threshold: float = 0.25,
    max_workers: int = 8,
) -> tuple[UnitMap, dict]:
    """Verify each extracted room and door crop in parallel using the Decisions API.

    Low-confidence items in [re_extract_threshold, confidence_threshold) are
    re-extracted against their cropped region; items below re_extract_threshold
    (or failing re-verification) are dropped.
    """
    with Image.open(image_path) as img:
        rgb_img = img.convert("RGB")
        room_crops = [(room, crop_room_region(rgb_img, room)) for room in unit_map.rooms]
        door_crops = [(door, crop_door_region(rgb_img, door)) for door in unit_map.doors]

    decision_client = _get_client(api_version="v1beta")
    neon_client: Optional[genai.Client] = None
    neon_lock = threading.Lock()

    def _get_neon_client() -> genai.Client:
        nonlocal neon_client
        with neon_lock:
            if neon_client is None:
                neon_client = _get_client()
            return neon_client

    def _verify_room(item: tuple[Room, Image.Image]) -> tuple[Optional[Room], Optional[float], bool]:
        room, crop = item
        b64_crop = _image_to_base64_png(crop)
        res = decision_client.decisions.create(
            model=DECISION_MODEL,
            input=[
                {"type": "image", "data": b64_crop, "mime_type": "image/png"},
                {
                    "type": "text",
                    "text": f"Cropped hospital floor plan room region. Candidate label: '{room.label}', type: '{room.type}'.",
                },
            ],
            questions={
                "is_room_label": {
                    "type": "noul",
                    "instructions": (
                        f"Is this a valid room or corridor region in the floor plan "
                        f"(tentatively '{room.label}')?"
                    ),
                }
            },
        )
        p_valid = float(res.answers["is_room_label"].noul)

        if p_valid >= confidence_threshold:
            return room.model_copy(update={"confidence": round(p_valid, 4)}), p_valid, False

        if p_valid >= re_extract_threshold:
            patched = _re_extract_room_label(crop, room, _get_neon_client())
            if patched is not None:
                re_res = decision_client.decisions.create(
                    model=DECISION_MODEL,
                    input=[
                        {"type": "image", "data": b64_crop, "mime_type": "image/png"},
                        {
                            "type": "text",
                            "text": f"Re-extracted room candidate: '{patched.label}' ({patched.type}).",
                        },
                    ],
                    questions={
                        "is_room_label": {
                            "type": "noul",
                            "instructions": f"Is this room labeled '{patched.label}'?",
                        }
                    },
                )
                re_p = float(re_res.answers["is_room_label"].noul)
                if re_p >= confidence_threshold:
                    return patched.model_copy(update={"confidence": round(re_p, 4)}), re_p, True

        return None, None, False

    def _verify_door(item: tuple[Door, Image.Image]) -> tuple[Optional[Door], Optional[float]]:
        door, crop = item
        b64_crop = _image_to_base64_png(crop)
        res = decision_client.decisions.create(
            model=DECISION_MODEL,
            input=[
                {"type": "image", "data": b64_crop, "mime_type": "image/png"},
                {
                    "type": "text",
                    "text": f"Cropped hospital floor plan doorway candidate '{door.id}' connecting {door.rooms_connected}.",
                },
            ],
            questions={
                "is_door": {
                    "type": "noul",
                    "instructions": "Is this a door or doorway connecting two spaces on the floor plan?",
                }
            },
        )
        p_door = float(res.answers["is_door"].noul)
        if p_door >= confidence_threshold:
            return door.model_copy(update={"confidence": round(p_door, 4)}), p_door
        return None, None

    kept_rooms: list[Room] = []
    kept_doors: list[Door] = []
    dropped = 0
    re_extracted = 0
    confidences: list[float] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        room_futures = [executor.submit(_verify_room, item) for item in room_crops]
        door_futures = [executor.submit(_verify_door, item) for item in door_crops]

        for fut in room_futures:
            verified_room, conf, was_re_extracted = fut.result()
            if verified_room is not None and conf is not None:
                kept_rooms.append(verified_room)
                confidences.append(conf)
                if was_re_extracted:
                    re_extracted += 1
            else:
                dropped += 1

        surviving_room_ids = {r.id for r in kept_rooms}

        for fut in door_futures:
            verified_door, conf = fut.result()
            if (
                verified_door is not None
                and conf is not None
                and all(rid in surviving_room_ids for rid in verified_door.rooms_connected)
            ):
                kept_doors.append(verified_door)
                confidences.append(conf)
            else:
                dropped += 1

    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    verified_unit = UnitMap(
        image_width=unit_map.image_width,
        image_height=unit_map.image_height,
        meters_per_pixel=unit_map.meters_per_pixel,
        rooms=kept_rooms,
        doors=kept_doors,
    )
    stats = {
        "kept": len(kept_rooms) + len(kept_doors),
        "dropped": dropped,
        "re_extracted": re_extracted,
        "mean_confidence": round(mean_conf, 4),
    }
    return verified_unit, stats


def load_fallback_unit(plan_id: Optional[str] = None) -> UnitMap:
    """Load the explicitly labeled pre-traced fallback unit map for a demo plan.

    When `plan_id` is provided (e.g. "icu_wing", "st_elizabeths_ward") and
    `data/demo/{plan_id}_fallback.json` exists, loads that plan's pre-traced
    fallback; otherwise loads `fallback_unit.json`.
    """
    target_path = FALLBACK_UNIT_PATH
    if plan_id:
        safe_id = plan_id.strip()
        if safe_id and "/" not in safe_id and "\\" not in safe_id and ".." not in safe_id:
            candidate = (
                ICU_WING_FALLBACK_PATH
                if safe_id == "icu_wing"
                else ST_ELIZABETHS_WARD_FALLBACK_PATH
                if safe_id == "st_elizabeths_ward"
                else DEMO_DIR / f"{safe_id}_fallback.json"
            )
            if candidate.is_file():
                target_path = candidate

    with open(target_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return UnitMap.model_validate(payload)
