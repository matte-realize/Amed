"""Custom robot designs for a specific unit (the Robot Lab).

Gemini 4 Carbon reads the unit's constraints (door widths, room types, tasks no product
covers yet) and proposes buildable robots with a bill of materials. Without an API key,
or if the call fails, a deterministic set of Amed designs sized to the unit is returned.
Costs are planning estimates, not quotes.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from pydantic import BaseModel, Field

from hospital_twin.schemas import UnitMap

logger = logging.getLogger(__name__)

CARBON_MODEL = "gemini-4-carbon"

BaseModelId = Literal[
    "amed_narrow_rover", "amed_prestage_pod", "amed_compounding_arm", "diligent_moxi", "aethon_tug"
]


class BomItem(BaseModel):
    part: str
    qty: int = Field(ge=1)
    unit_cost_usd: float = Field(ge=0)
    source: Literal["off_the_shelf", "custom_fabricated", "custom_molded"]
    notes: str = ""


class RobotDesign(BaseModel):
    id: str
    name: str
    tagline: str
    base_model: BaseModelId
    rooms_served: list[str]
    tasks: list[str]
    why_custom: str
    width_m: float
    length_m: float
    height_m: float
    payload_kg: float
    speed_mps: float
    min_door_m: float
    bom: list[BomItem]
    assembly_hours: float
    tooling_cost_usd: float
    buildability_notes: list[str]
    standards: list[str]
    risks: list[str]


class RobotDesignReport(BaseModel):
    source: Literal["gemini", "fallback"]
    model: str = ""
    constraints: dict[str, float | int | str]
    designs: list[RobotDesign]


class _DesignOutput(BaseModel):
    designs: list[RobotDesign]


def unit_constraints(unit: UnitMap) -> dict[str, float | int | str]:
    widths = sorted(d.width_m for d in unit.doors if d.width_m)
    types = sorted({r.type for r in unit.rooms})
    return {
        "rooms": len(unit.rooms),
        "doors": len(unit.doors),
        "narrowest_door_m": round(widths[0], 2) if widths else 0.9,
        "doors_under_0_90_m": sum(1 for w in widths if w < 0.9),
        "room_types": ", ".join(types),
    }


def _fallback_designs(c: dict[str, float | int | str]) -> list[RobotDesign]:
    narrow = float(c["narrowest_door_m"])
    rover_w = round(max(0.48, min(0.62, narrow - 0.14)), 2)
    oem = "off_the_shelf"
    return [
        RobotDesign(
            id="amed_slim_rover",
            name="Amed Slim Rover",
            tagline=f"A {rover_w:.2f} m courier that clears this unit's {narrow:.2f} m doors.",
            base_model="amed_narrow_rover",
            rooms_served=["patient_room", "medication_room", "supply_room", "lab_drop"],
            tasks=["Drop meds and supplies at patient doors", "Carry specimens to the lab drop", "Run restock trips overnight"],
            why_custom="Commercial couriers need about 0.90 m of door clearance; older wings are narrower.",
            width_m=rover_w, length_m=0.78, height_m=1.05, payload_kg=30, speed_mps=0.85, min_door_m=round(rover_w + 0.1, 2),
            bom=[
                BomItem(part="2D safety LiDAR", qty=1, unit_cost_usd=450, source=oem),
                BomItem(part="Depth cameras", qty=2, unit_cost_usd=300, source=oem),
                BomItem(part="Hub motors with encoders", qty=2, unit_cost_usd=350, source=oem),
                BomItem(part="Motor controller", qty=1, unit_cost_usd=250, source=oem),
                BomItem(part="Edge compute module", qty=1, unit_cost_usd=700, source=oem),
                BomItem(part="LiFePO4 battery pack, 48 V", qty=1, unit_cost_usd=600, source=oem),
                BomItem(part="Laser-cut aluminum chassis", qty=1, unit_cost_usd=900, source="custom_fabricated"),
                BomItem(part="Locking bins with badge reader", qty=2, unit_cost_usd=250, source="custom_fabricated"),
                BomItem(part="Body panels (thermoformed, molded at volume)", qty=1, unit_cost_usd=800, source="custom_molded", notes="Wipe-down, disinfectant-safe plastic"),
                BomItem(part="Bumpers, e-stop, safety relays", qty=1, unit_cost_usd=200, source=oem),
                BomItem(part="Wi-Fi/LTE and elevator interface", qty=1, unit_cost_usd=350, source=oem),
                BomItem(part="Wiring harness and fasteners", qty=1, unit_cost_usd=300, source=oem),
            ],
            assembly_hours=18, tooling_cost_usd=120_000,
            buildability_notes=[
                "Drivetrain, sensors and compute are catalog parts; only the chassis, bins and panels are custom.",
                "Chassis is flat-pattern sheet metal, so any local fab shop can make prototypes.",
                "Panels start thermoformed; an injection mold pays off after a few hundred units.",
            ],
            standards=["UL 3300 (service robots)", "ISO 13482 (safety of service robots)", "IEC 62133 (batteries)", "Hospital infection-control review"],
            risks=["Narrow body limits payload and tip stability; needs a low battery mount.", "Elevator integration varies by building."],
        ),
        RobotDesign(
            id="amed_emergency_cart",
            name="Amed Emergency Cart",
            tagline="Pre-stages code supplies at the right room before the patient arrives.",
            base_model="amed_prestage_pod",
            rooms_served=["corridor", "supply_room", "patient_room"],
            tasks=["Bring the crash cart to a remote room", "Stage blood and code meds on an EMS alert", "Check its own stock and battery every shift"],
            why_custom="Moving and checking the crash cart is still done by hand; no product covers it.",
            width_m=0.72, length_m=1.05, height_m=1.1, payload_kg=120, speed_mps=1.1, min_door_m=0.85,
            bom=[
                BomItem(part="2D safety LiDAR", qty=2, unit_cost_usd=450, source=oem),
                BomItem(part="Depth cameras", qty=3, unit_cost_usd=300, source=oem),
                BomItem(part="Drive motors with gearboxes", qty=2, unit_cost_usd=600, source=oem),
                BomItem(part="Motor controller", qty=1, unit_cost_usd=400, source=oem),
                BomItem(part="Edge compute module", qty=1, unit_cost_usd=700, source=oem),
                BomItem(part="Battery pack, 48 V high capacity", qty=1, unit_cost_usd=1200, source=oem),
                BomItem(part="Welded steel and aluminum frame", qty=1, unit_cost_usd=1600, source="custom_fabricated"),
                BomItem(part="Locking drawers with RFID tags", qty=4, unit_cost_usd=350, source="custom_fabricated"),
                BomItem(part="RFID reader and antennas", qty=1, unit_cost_usd=900, source=oem, notes="Counts drawer contents and expiry"),
                BomItem(part="Body panels", qty=1, unit_cost_usd=1500, source="custom_molded"),
                BomItem(part="Bumpers, e-stop, beacon", qty=1, unit_cost_usd=300, source=oem),
                BomItem(part="Comms and wiring", qty=1, unit_cost_usd=750, source=oem),
            ],
            assembly_hours=30, tooling_cost_usd=180_000,
            buildability_notes=[
                "Drawer module reuses standard crash-cart drawer sizes, so hospitals keep their layouts.",
                "RFID stock check uses catalog readers; the work is in tag placement and software.",
                "Frame is welded in prototypes and moves to bent sheet metal for volume.",
            ],
            standards=["UL 3300", "ISO 13482", "IEC 62133", "Joint Commission crash-cart check policy"],
            risks=["Must never block an egress path during a code.", "Clinical sign-off needed for what it may carry."],
        ),
        RobotDesign(
            id="amed_compounding_cell",
            name="Amed Compounding Cell",
            tagline="A 6-axis arm in a clean hood that draws up and checks IV doses.",
            base_model="amed_compounding_arm",
            rooms_served=["medication_room"],
            tasks=["Draw up and compound IV doses", "Weigh-check every dose", "Log each dose for the pharmacist"],
            why_custom="Commercial IV robots are sized for central pharmacies; this fits a unit med room.",
            width_m=1.2, length_m=0.8, height_m=1.9, payload_kg=5, speed_mps=0.0, min_door_m=0.9,
            bom=[
                BomItem(part="6-axis collaborative arm", qty=1, unit_cost_usd=28000, source=oem),
                BomItem(part="Syringe and vial gripper", qty=1, unit_cost_usd=4500, source=oem),
                BomItem(part="Vision camera and barcode reader", qty=1, unit_cost_usd=1500, source=oem),
                BomItem(part="Precision scale for weight checks", qty=1, unit_cost_usd=3000, source=oem),
                BomItem(part="Laminar-flow hood enclosure", qty=1, unit_cost_usd=12000, source=oem),
                BomItem(part="Syringe and vial fixtures", qty=1, unit_cost_usd=2500, source="custom_fabricated"),
                BomItem(part="Cell frame and work surface", qty=1, unit_cost_usd=2000, source="custom_fabricated"),
                BomItem(part="Controller PC", qty=1, unit_cost_usd=1200, source=oem),
                BomItem(part="Door interlocks and light curtain", qty=1, unit_cost_usd=800, source=oem),
            ],
            assembly_hours=40, tooling_cost_usd=40_000,
            buildability_notes=[
                "Most of the cost is the arm and hood, both bought; custom work is fixtures and software.",
                "Fixtures are machined at low volume, then cast or molded.",
                "Fits through a 0.90 m door when the hood is shipped separately.",
            ],
            standards=["USP <797> sterile compounding", "ISO 10218 / ISO/TS 15066 (collaborative robots)", "IEC 61010-1"],
            risks=["Pharmacy validation is slow; plan a supervised pilot.", "Arm cost dominates until volume pricing."],
        ),
    ]


def _design_with_carbon(c: dict[str, float | int | str]) -> list[RobotDesign]:
    from google import genai
    from google.genai import types

    client = genai.Client(http_options=types.HttpOptions(timeout=60_000))
    prompt = (
        "You are a hospital robotics engineer. Propose 3 custom robots for ONE hospital unit with these constraints:\n"
        f"{c}\n"
        "Each robot must fit through the narrowest door listed (min_door_m <= narrowest_door_m unless it is a fixed "
        "in-room cell), take over repetitive physical work in specific room types (medication_room, supply_room, "
        "lab_drop, patient_room, corridor, nurse_station), and leave clinical judgment to staff.\n"
        "For each: a realistic bill of materials with mostly off-the-shelf parts and conservative 2026 USD prototype "
        "prices, assembly hours, tooling cost for volume production, buildability notes, the safety standards to check, "
        "and the main risks. base_model must be the closest of: amed_narrow_rover, amed_prestage_pod, "
        "amed_compounding_arm, diligent_moxi, aethon_tug. Use plain language a hospital manager understands."
    )
    response = client.models.generate_content(
        model=CARBON_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=_DesignOutput),
    )
    return _DesignOutput.model_validate_json(response.text).designs[:3]


def design_custom_robots(unit: UnitMap, live: bool = True) -> RobotDesignReport:
    c = unit_constraints(unit)
    if live and os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            designs = _design_with_carbon(c)
            if designs:
                return RobotDesignReport(source="gemini", model=CARBON_MODEL, constraints=c, designs=designs)
        except Exception as exc:  # network, quota or schema errors fall back to the sized Amed designs
            logger.warning("Gemini 4 Carbon robot design failed, using fallback: %s", exc)
    return RobotDesignReport(source="fallback", constraints=c, designs=_fallback_designs(c))


class DesignBrief(BaseModel):
    """An engineer's scenario: what the robot must do and the limits it must meet."""

    scenario: str = Field(min_length=3, max_length=2000)
    max_width_m: float = Field(default=0.7, gt=0.2, le=2.0)
    payload_kg: float = Field(default=30, ge=0, le=500)
    target_unit_cost_usd: float = Field(default=15000, ge=0)
    rooms: list[str] = Field(default_factory=list)


class BriefDesignResult(BaseModel):
    source: Literal["gemini", "draft"]
    model: str = ""
    design: RobotDesign


def _pick_base(scenario: str) -> str:
    t = scenario.lower()
    if any(k in t for k in ("compound", "iv ", "syringe", "vial", "pharmac", "arm", "pick", "sort", "lab bench")):
        return "amed_compounding_cell"
    if any(k in t for k in ("crash", "code", "emergency", "blood", "trauma", "surge", "cart", "stage")):
        return "amed_emergency_cart"
    return "amed_slim_rover"


def _draft_from_brief(unit: UnitMap, brief: DesignBrief) -> RobotDesign:
    """Deterministic draft: closest Amed reference design, resized to the brief's limits."""
    c = unit_constraints(unit)
    base_id = _pick_base(brief.scenario)
    base = next(d for d in _fallback_designs(c) if d.id == base_id)
    mobile = base.base_model != "amed_compounding_arm"
    width = min(base.width_m, brief.max_width_m - 0.08) if mobile else base.width_m
    first = brief.scenario.strip().rstrip(".").split(".")[0][:140]
    payload_scale = max(0.6, min(2.5, (brief.payload_kg or base.payload_kg) / max(base.payload_kg, 1)))
    bom = [
        b.model_copy(update={"unit_cost_usd": round(b.unit_cost_usd * (payload_scale if "motor" in b.part.lower() or "battery" in b.part.lower() or "frame" in b.part.lower() or "chassis" in b.part.lower() else 1), 0)})
        for b in base.bom
    ]
    return base.model_copy(update={
        "id": "brief_" + base_id,
        "name": f"Custom {base.name.replace('Amed ', '')}",
        "tagline": first + ".",
        "rooms_served": brief.rooms or base.rooms_served,
        "tasks": [first] + base.tasks[:2],
        "why_custom": f"Drafted from your brief: {brief.max_width_m:.2f} m door limit, {brief.payload_kg:.0f} kg payload.",
        "width_m": round(max(0.4, width), 2),
        "min_door_m": round(max(0.4, width) + 0.08, 2) if mobile else base.min_door_m,
        "payload_kg": brief.payload_kg or base.payload_kg,
        "bom": bom,
    })


def design_from_brief(unit: UnitMap, brief: DesignBrief) -> BriefDesignResult:
    if os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(http_options=types.HttpOptions(timeout=45_000))
            prompt = (
                "You are a hospital robotics engineer. Design ONE robot for this scenario, written by an engineer:\n"
                f"\"{brief.scenario}\"\n"
                f"Hard limits: overall width <= {brief.max_width_m} m (it must pass these doors), payload >= {brief.payload_kg} kg, "
                f"target unit cost at 1,000 units about ${brief.target_unit_cost_usd:,.0f}. Rooms it serves: {brief.rooms or 'any'}.\n"
                f"Unit constraints: {unit_constraints(unit)}\n"
                "Give a realistic bill of materials with mostly off-the-shelf parts and conservative 2026 USD prototype prices, "
                "assembly hours, tooling cost for volume, buildability notes, safety standards to check and main risks. "
                "base_model must be the closest of: amed_narrow_rover, amed_prestage_pod, amed_compounding_arm, diligent_moxi, aethon_tug."
            )
            response = client.models.generate_content(
                model=CARBON_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=RobotDesign),
            )
            design = RobotDesign.model_validate_json(response.text)
            return BriefDesignResult(source="gemini", model=CARBON_MODEL, design=design)
        except Exception as exc:
            logger.warning("Gemini 4 Carbon brief design failed, using draft: %s", exc)
    return BriefDesignResult(source="draft", design=_draft_from_brief(unit, brief))
