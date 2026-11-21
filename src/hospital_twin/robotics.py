"""Catalog of hospital courier robot profiles and evidence-backed hospital struggles."""

from __future__ import annotations

from hospital_twin.schemas import RobotProfile, RunType

ALL_RUN_TYPES: list[RunType] = [
    "pharmacy_pickup",
    "lab_specimen_drop",
    "supply_restock",
]

ROBOT_CATALOG: dict[str, RobotProfile] = {
    "aethon_tug": RobotProfile(
        id="aethon_tug",
        name="Aethon TUG + 3-Compartment Pod",
        vendor="Aethon",
        is_custom=False,
        speed_mps=0.76,
        clearance_m=0.90,
        length_m=0.965,
        compartments=3,
        handoff_seconds=20.0,
        supported_runs=list(ALL_RUN_TYPES),
        solves_problem="High-volume routine medication, specimen, and supply runs (~10,000 med orders/week)",
        description=(
            "Autonomous mobile courier with 3-compartment locked custodial pod "
            "and 20s badge-and-unlock handoffs."
        ),
    ),
    "diligent_moxi": RobotProfile(
        id="diligent_moxi",
        name="Diligent Moxi Cobot",
        vendor="Diligent Robotics",
        is_custom=False,
        speed_mps=0.90,
        clearance_m=0.85,
        length_m=0.75,
        compartments=1,
        handoff_seconds=12.0,
        supported_runs=list(ALL_RUN_TYPES),
        solves_problem="Point-to-point nurse-station fetch and lab/pharmacy errands with manipulator arm",
        description=(
            "Socially intelligent hospital cobot capable of elevator/door interaction "
            "and rapid 12s handoffs."
        ),
    ),
    "pneumatic_tube": RobotProfile(
        id="pneumatic_tube",
        name="Pneumatic Tube + Runner Hybrid",
        vendor="Swisslog / Facility Infrastructure",
        is_custom=False,
        speed_mps=3.50,
        clearance_m=0.0,
        length_m=0.30,
        compartments=1,
        handoff_seconds=10.0,
        supported_runs=["lab_specimen_drop", "pharmacy_pickup"],
        solves_problem="Ultra-fast STAT lab specimen and small-dose pharmacy transport bypassing corridor congestion",
        description=(
            "High-speed pneumatic tube carrier for specimens and medications; "
            "bulky supply restocks fall back to nurses."
        ),
    ),
    "amed_prestage_pod": RobotProfile(
        id="amed_prestage_pod",
        name="Amed Rapid Pre-Stage Cart",
        vendor="Amed Custom",
        is_custom=True,
        speed_mps=1.10,
        clearance_m=0.90,
        length_m=1.05,
        compartments=4,
        handoff_seconds=10.0,
        supported_runs=list(ALL_RUN_TYPES),
        solves_problem="Delayed defibrillation, missing crash-cart supplies, and EMS surge pre-staging",
        description=(
            "Custom high-speed 4-compartment rapid-response cart with RFID-verified "
            "pre-staging and 10s quick-release handoff."
        ),
    ),
    "amed_narrow_rover": RobotProfile(
        id="amed_narrow_rover",
        name="Amed Slim-Chassis Rover",
        vendor="Amed Custom",
        is_custom=True,
        speed_mps=0.85,
        clearance_m=0.72,
        length_m=0.70,
        compartments=2,
        handoff_seconds=15.0,
        supported_runs=list(ALL_RUN_TYPES),
        solves_problem="Legacy wings with narrow (<0.90m) doorways and contamination-prone corridor bottlenecks",
        description=(
            "Custom narrow-chassis autonomous courier engineered for 0.72m doorway "
            "clearance and sealed dual-compartment transport."
        ),
    ),
}

HOSPITAL_STRUGGLES: list[dict[str, str]] = [
    {
        "id": "delayed_defibrillation",
        "title": "Delayed Defibrillation & Missing Crash-Cart Supplies",
        "stat": "30.1% of in-hospital cardiac arrests experienced delayed defibrillation (>2 min)",
        "source": "In-hospital cardiac arrest resuscitation study (delayed defibrillation cohort)",
        "recommended_scenario": "crash_cart_missing",
        "recommended_robot": "amed_prestage_pod",
        "run_description": (
            "supply restock from supply room to nurse station every 15 minutes; "
            "pharmacy pickup from medication room to nurse station every 20 minutes"
        ),
    },
    {
        "id": "pediatric_epinephrine_delay",
        "title": "Time-Critical Medication & Epinephrine Delivery Delays",
        "stat": "0.95 survival risk ratio per minute of epinephrine delay in pediatric arrest",
        "source": "PMC6191294 (Pediatric in-hospital cardiac arrest epinephrine timing)",
        "recommended_scenario": "pharmacy_delay",
        "recommended_robot": "pneumatic_tube",
        "run_description": (
            "pharmacy pickup from medication room to nurse station every 12 minutes; "
            "lab specimen drop from patient room to lab drop every 20 minutes"
        ),
    },
    {
        "id": "malpractice_communication_handoff",
        "title": "Handoff Breakdowns & Custody Tracking Gaps",
        "stat": "28.6% of malpractice claims involve communication/handoff failures ($661,000 average indemnity)",
        "source": "Coverys 2025 Malpractice Claims Analysis",
        "recommended_scenario": "staffing_gap",
        "recommended_robot": "aethon_tug",
        "run_description": (
            "pharmacy pickup from medication room to nurse station every 15 minutes; "
            "lab specimen drop from nurse station to lab drop every 20 minutes; "
            "supply restock from supply room to nurse station every 30 minutes"
        ),
    },
    {
        "id": "resistant_organism_outbreaks",
        "title": "Contamination & Narrow-Corridor Access Bottlenecks",
        "stat": "100% (23/23) of studied hospital outbreaks linked to resistant-organism drain/wet-zone reservoirs",
        "source": "Hospital drain-associated resistant-organism outbreak review (23/23 studies)",
        "recommended_scenario": "chaotic",
        "recommended_robot": "amed_narrow_rover",
        "run_description": (
            "lab specimen drop from patient room to lab drop every 15 minutes; "
            "supply restock from supply room to nurse station every 25 minutes"
        ),
    },
    {
        "id": "high_volume_med_orders",
        "title": "High-Volume Medication & Surge Order Overload",
        "stat": "~10,000 medication orders/week at a typical 200-bed hospital",
        "source": "Aethon via RIA (Robotics Industry Association)",
        "recommended_scenario": "surge",
        "recommended_robot": "diligent_moxi",
        "run_description": (
            "pharmacy pickup from medication room to nurse station every 10 minutes; "
            "lab specimen drop from nurse station to lab drop every 15 minutes; "
            "supply restock from supply room to nurse station every 20 minutes"
        ),
    },
]


def get_robot_profile(robot_profile_id: str) -> RobotProfile:
    """Return the matching RobotProfile, defaulting to aethon_tug if unknown."""
    return ROBOT_CATALOG.get(robot_profile_id, ROBOT_CATALOG["aethon_tug"])
