"""Turn an unrehearsed "what if" prompt into simulation settings.

Gemini 4 Hydrogen maps free text (anything a charge nurse or emergency manager might ask)
onto the engine's scenario modes, fleet and run schedule. A keyword fallback keeps it
working offline. The result always names how the prompt was understood, so the user can
see and correct the interpretation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

HYDROGEN_MODEL = "gemini-4-hydrogen"

ScenarioMode = Literal[
    "routine", "chaotic", "surge", "robot_failure", "pharmacy_delay", "staffing_gap",
    "crash_cart_missing", "cardiac_arrest", "contamination_quarantine", "catastrophic_mci",
]
RobotId = Literal[
    "aethon_tug", "diligent_moxi", "pneumatic_tube", "amed_prestage_pod", "amed_narrow_rover",
    "amed_compounding_arm", "amed_surgical_prep_bot", "mixed_fleet",
]


class ScenarioSettings(BaseModel):
    scenario_mode: ScenarioMode
    num_robots: int = Field(ge=1, le=3)
    robot_profile_id: RobotId
    ems_preposition: bool
    shift_hours: int = Field(ge=1, le=24, description="Shift length in hours: 4, 8 or 12.")
    run_description: str = Field(min_length=5)
    understood_as: str = Field(description="One plain sentence describing how the prompt was interpreted.")


class ScenarioResult(BaseModel):
    source: Literal["gemini", "keywords"]
    settings: ScenarioSettings


_RULES: list[tuple[tuple[str, ...], ScenarioMode]] = [
    (("mass casualty", "mass-casualty", "mci", "bus crash", "pile-up", "pileup", "shooting", "explosion", "disaster"), "catastrophic_mci"),
    (("cardiac arrest", "code blue", "arrest", "crash cart needed", "defib"), "cardiac_arrest"),
    (("contamina", "outbreak", "quarantine", "infection", "c. diff", "mrsa", "isolation breach"), "contamination_quarantine"),
    (("robot broke", "robot down", "robot fail", "robot is broken", "breakdown", "breaks down", "broken", "stops working", "battery dies", "robot outage"), "robot_failure"),
    (("pharmacy", "med delay", "medication delay", "pharmacist"), "pharmacy_delay"),
    (("short staff", "staffing", "sick call", "call out", "call in sick", "calls in sick", "sick", "only one nurse", "understaff", "nurse out"), "staffing_gap"),
    (("crash cart missing", "cart missing", "cart not ready", "expired", "dead battery"), "crash_cart_missing"),
    (("surge", "flu", "influx", "busy", "overflow", "ambulance"), "surge"),
    (("quiet", "normal", "routine", "calm", "weekend"), "routine"),
]


def _keyword_settings(prompt: str) -> ScenarioSettings:
    t = prompt.lower()
    mode: ScenarioMode = "chaotic"
    for keys, m in _RULES:
        if any(k in t for k in keys):
            mode = m
            break
    robots = 2
    m = re.search(r"\b([1-3]|one|two|three)\s+robots?\b", t)
    if m:
        robots = {"one": 1, "two": 2, "three": 3}.get(m.group(1), None) or int(m.group(1))
    elif mode in ("catastrophic_mci", "surge"):
        robots = 3
    robot: RobotId = "aethon_tug"
    if "narrow" in t or "old wing" in t or "tight" in t:
        robot = "amed_narrow_rover"
    elif mode in ("catastrophic_mci", "cardiac_arrest", "crash_cart_missing"):
        robot = "amed_prestage_pod"
    elif "mixed" in t or "all robots" in t:
        robot = "mixed_fleet"
    hours: int = 12 if ("night shift" in t or re.search(r"\b12[- ]?hour", t)) else 4 if re.search(r"\b4[- ]?hour", t) else 8
    ems = mode in ("catastrophic_mci", "surge", "cardiac_arrest") or "ambulance" in t or "ems" in t
    busy = mode in ("catastrophic_mci", "surge", "cardiac_arrest", "chaotic")
    runs = (
        "STAT crash cart pre-stage every 15 minutes, pharmacy pickup every 15 minutes, lab specimen drop every 20 minutes, supply restock every 45 minutes"
        if busy else
        "pharmacy pickup every 30 minutes, lab specimen drop every 45 minutes, supply restock every 2 hours"
    )
    label = {
        "catastrophic_mci": "mass-casualty", "chaotic": "busy", "surge": "patient-surge", "robot_failure": "robot-breakdown",
        "pharmacy_delay": "pharmacy-delay", "staffing_gap": "short-staffed", "crash_cart_missing": "crash-cart-not-ready",
        "cardiac_arrest": "cardiac-arrest", "contamination_quarantine": "contamination", "routine": "routine",
    }.get(mode, mode.replace("_", " "))
    return ScenarioSettings(
        scenario_mode=mode, num_robots=robots, robot_profile_id=robot, ems_preposition=ems,
        shift_hours=hours, run_description=runs,
        understood_as=f"A {label} shift of {hours} hours with {robots} robot{'s' if robots > 1 else ''}"
        + (", using ambulance pre-alerts" if ems else "") + ".",
    )


def scenario_from_prompt(prompt: str) -> ScenarioResult:
    prompt = (prompt or "").strip()[:1500]
    if os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(http_options=types.HttpOptions(timeout=20_000))
            instruction = (
                "You set up a hospital unit simulation from a free-text 'what if' question. Map it to the closest "
                "settings. scenario_mode: routine, chaotic (busy day with urgent orders), surge, robot_failure, "
                "pharmacy_delay, staffing_gap, crash_cart_missing, cardiac_arrest, contamination_quarantine, "
                "catastrophic_mci. robot_profile_id: aethon_tug (standard courier), diligent_moxi, pneumatic_tube "
                "(small items only), amed_prestage_pod (emergency cart), amed_narrow_rover (narrow doors), "
                "amed_compounding_arm, amed_surgical_prep_bot, mixed_fleet. run_description lists delivery runs in "
                "plain English with frequencies, e.g. 'pharmacy pickup every 20 minutes, lab specimen drop every 30 "
                "minutes, supply restock hourly'; make it heavier for emergencies. understood_as is one plain "
                "sentence for a hospital manager, no jargon.\n\nQuestion: " + prompt
            )
            response = client.models.generate_content(
                model=HYDROGEN_MODEL,
                contents=instruction,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=ScenarioSettings),
            )
            settings = ScenarioSettings.model_validate_json(response.text)
            settings.shift_hours = min((4, 8, 12), key=lambda h: abs(h - settings.shift_hours))
            return ScenarioResult(source="gemini", settings=settings)
        except Exception as exc:
            logger.warning("Gemini 4 Hydrogen scenario parse failed, using keywords: %s", exc)
    return ScenarioResult(source="keywords", settings=_keyword_settings(prompt))
