"""Natural language shift run schedule parser using Gemini 4 Hydrogen with regex fallback."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from hospital_twin.schemas import RunType, ScheduledRun

logger = logging.getLogger(__name__)

HYDROGEN_MODEL_ID = "gemini-4-hydrogen"
DEFAULT_SHIFT_MINUTES = 480.0


class _ScheduleOutput(BaseModel):
    runs: list[ScheduledRun] = Field(default_factory=list)


_NURSE_KEYWORDS = ("nurse", "station", "ns", "hub", "desk")
_PHARMACY_KEYWORDS = ("med", "pharm", "rx", "dispens", "drug")
_LAB_KEYWORDS = ("lab", "spec", "path", "blood")
_SUPPLY_KEYWORDS = ("supply", "stor", "closet", "linen", "central")
_PATIENT_KEYWORDS = ("patient", "room", "ward", "bed")


def _pick_room(
    available_rooms: list[str],
    keywords: tuple[str, ...],
    exclude: Optional[str] = None,
) -> str:
    """Pick the best matching room ID from available_rooms."""
    if not available_rooms:
        raise ValueError("available_rooms must not be empty.")

    candidates = [r for r in available_rooms if r != exclude] or list(available_rooms)
    for kw in keywords:
        for room_id in candidates:
            if kw in room_id.lower():
                return room_id

    non_corridor = [r for r in candidates if "corridor" not in r.lower() and "hall" not in r.lower()]
    return non_corridor[0] if non_corridor else candidates[0]


def _resolve_room(room_id: str, available_rooms: list[str], fallback_keywords: tuple[str, ...], exclude: Optional[str] = None) -> str:
    """Resolve a room ID against available_rooms (exact, case-insensitive, or keyword fallback)."""
    if room_id in available_rooms and room_id != exclude:
        return room_id
    lower_map = {r.lower(): r for r in available_rooms}
    if room_id.lower() in lower_map and lower_map[room_id.lower()] != exclude:
        return lower_map[room_id.lower()]
    return _pick_room(available_rooms, fallback_keywords, exclude=exclude)


def _default_endpoints(run_type: RunType, available_rooms: list[str]) -> tuple[str, str]:
    nurse_station = _pick_room(available_rooms, _NURSE_KEYWORDS)
    if run_type == "pharmacy_pickup":
        origin = _pick_room(available_rooms, _PHARMACY_KEYWORDS, exclude=nurse_station)
        dest = _resolve_room(nurse_station, available_rooms, _NURSE_KEYWORDS, exclude=origin)
        return origin, dest
    if run_type == "lab_specimen_drop":
        dest = _pick_room(available_rooms, _LAB_KEYWORDS, exclude=nurse_station)
        origin = _resolve_room(nurse_station, available_rooms, _NURSE_KEYWORDS, exclude=dest)
        return origin, dest
    # supply_restock
    origin = _pick_room(available_rooms, _SUPPLY_KEYWORDS, exclude=nurse_station)
    dest = _resolve_room(nurse_station, available_rooms, _PATIENT_KEYWORDS + _NURSE_KEYWORDS, exclude=origin)
    return origin, dest


def _extract_frequency_minutes(clause: str, default_minutes: float) -> float:
    c = clause.lower()

    if "half hour" in c or "half-hour" in c:
        return 30.0
    if "hourly" in c or "every hour" in c or "once an hour" in c or "1x/hr" in c:
        return 60.0

    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes|m\b)", c)
    if m:
        return max(1.0, float(m.group(1)))

    m = re.search(r"every\s+(\d+(?:\.\d+)?)\s*(?:hr|hrs|hour|hours|h\b)", c)
    if m:
        return max(1.0, float(m.group(1)) * 60.0)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\s*(?:a|an|per)\s*hour", c)
    if m:
        count = max(1.0, float(m.group(1)))
        return max(1.0, 60.0 / count)

    if "twice an hour" in c or "twice per hour" in c:
        return 30.0

    if "twice a shift" in c or "twice per shift" in c or "twice" in c:
        return DEFAULT_SHIFT_MINUTES / 2.0
    if "once a shift" in c or "once per shift" in c:
        return DEFAULT_SHIFT_MINUTES

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\s*(?:a|per)?\s*shift", c)
    if m:
        count = max(1.0, float(m.group(1)))
        return max(1.0, DEFAULT_SHIFT_MINUTES / count)

    return default_minutes


def _detect_run_type(clause: str) -> Optional[RunType]:
    c = clause.lower()
    if any(k in c for k in ("pharm", "med", "rx", "drug", "pill")):
        return "pharmacy_pickup"
    if any(k in c for k in ("specimen", "lab", "blood", "sample", "urine", "path")):
        return "lab_specimen_drop"
    if any(k in c for k in ("supply", "restock", "linen", "stock", "gauze", "iv")):
        return "supply_restock"
    return None


def _fallback_parse(text: str, available_rooms: list[str]) -> list[ScheduledRun]:
    """Deterministic regex/rule-based parser used when GEMINI_API_KEY is absent."""
    if not text or not text.strip() or not available_rooms:
        return []

    clauses = [c.strip() for c in re.split(r"[,;\n]+|\band\b", text) if c.strip()]
    runs: list[ScheduledRun] = []

    default_freq: dict[RunType, float] = {
        "pharmacy_pickup": 30.0,
        "lab_specimen_drop": 60.0,
        "supply_restock": 240.0,
    }
    labels: dict[RunType, str] = {
        "pharmacy_pickup": "Pharmacy pickup",
        "lab_specimen_drop": "Lab specimen drop",
        "supply_restock": "Supply restock",
    }

    for clause in clauses:
        run_type = _detect_run_type(clause)
        if run_type is None:
            # Check if clause still expresses a frequency (e.g. "deliveries every 45 minutes").
            freq = _extract_frequency_minutes(clause, -1.0)
            if freq <= 0:
                continue
            run_type = "supply_restock"
        else:
            freq = _extract_frequency_minutes(clause, default_freq[run_type])

        origin, dest = _default_endpoints(run_type, available_rooms)
        runs.append(
            ScheduledRun(
                run_type=run_type,
                origin=origin,
                destination=dest,
                frequency_minutes=freq,
                label=labels[run_type],
            )
        )

    if not runs:
        # Generic non-empty input with no recognized keywords: emit one conservative default run.
        origin, dest = _default_endpoints("supply_restock", available_rooms)
        runs.append(
            ScheduledRun(
                run_type="supply_restock",
                origin=origin,
                destination=dest,
                frequency_minutes=60.0,
                label="Supply restock",
            )
        )

    return runs


def _parse_with_hydrogen(text: str, available_rooms: list[str]) -> list[ScheduledRun]:
    from google import genai
    from google.genai import types

    client = genai.Client()
    prompt = (
        "Convert the following hospital unit shift run description into a structured list of "
        "ScheduledRun objects.\n"
        f"Available room IDs: {available_rooms}\n"
        "Rules:\n"
        "- run_type must be one of: pharmacy_pickup, lab_specimen_drop, supply_restock.\n"
        "- origin and destination must each be one of the available room IDs, and different if possible.\n"
        "- frequency_minutes must be a positive number (assume an 8-hour = 480-minute shift for 'twice a shift', etc.).\n\n"
        f"Description: {text}"
    )

    response = client.models.generate_content(
        model=HYDROGEN_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_ScheduleOutput,
        ),
    )

    parsed = _ScheduleOutput.model_validate_json(response.text)
    normalized: list[ScheduledRun] = []
    for run in parsed.runs:
        default_origin, default_dest = _default_endpoints(run.run_type, available_rooms)
        origin = _resolve_room(run.origin, available_rooms, (default_origin,))
        dest = _resolve_room(run.destination, available_rooms, (default_dest,), exclude=origin)
        normalized.append(
            ScheduledRun(
                run_type=run.run_type,
                origin=origin,
                destination=dest,
                frequency_minutes=max(1.0, float(run.frequency_minutes)),
                label=run.label,
            )
        )
    return normalized


def parse_run_description(text: str, available_rooms: list[str]) -> list[ScheduledRun]:
    """Parse a natural language run description into ScheduledRun objects.

    Uses Gemini 4 Hydrogen structured JSON output when ``GEMINI_API_KEY`` is set,
    falling back deterministically to regex/rule parsing otherwise.
    """
    if not text or not text.strip() or not available_rooms:
        return []

    if os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            runs = _parse_with_hydrogen(text, available_rooms)
            if runs:
                return runs
        except Exception as exc:
            logger.warning("Gemini 4 Hydrogen parsing failed, using fallback: %s", exc)

    return _fallback_parse(text, available_rooms)
