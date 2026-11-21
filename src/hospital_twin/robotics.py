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
        dimensions="96.5 cm L × 56 cm W × 112 cm H",
        cost_estimate_usd="~$4,500/mo lease or $120k–$160k unit",
        design_notes=(
            "Low-profile differential-drive AMR base with 270° LiDAR, "
            "top-mounted 3-compartment custodial pod, and badge-actuated locker doors."
        ),
        why_helpful=(
            "Eliminates 20–35 min of routine nurse walking per shift while "
            "maintaining chain-of-custody logs."
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
        dimensions="75 cm L × 52 cm W × 148 cm H",
        cost_estimate_usd="~$3,500–$4,500/mo RaaS",
        design_notes=(
            "Slim omnidirectional mobile base with vertical torso column, "
            "7-DOF cobot arm for badge/elevator/door interaction, and social LED head."
        ),
        why_helpful=(
            "Interacts with existing badge doors and supply bins without "
            "costly facility retrofits."
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
        dimensions="110 mm / 160 mm carrier tube",
        cost_estimate_usd="~$15k–$35k per station drop",
        design_notes=(
            "Pressurized air-tube network between Nurse Station, Lab Drop, "
            "and Pharmacy with soft-landing diverters."
        ),
        why_helpful=(
            "Cuts urgent specimen and stat medication transit from minutes to "
            "seconds; bulky supplies remain manual."
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
        dimensions="105 cm L × 62 cm W × 118 cm H",
        cost_estimate_usd="~$28k BOM target (custom modular cart)",
        design_notes=(
            "Custom 4-bay crash-cart and blood-cooler staging chassis built on "
            "commercial AMR drive with amber priority beacon and auto-latching dock."
        ),
        why_helpful=(
            "Pre-positions emergency equipment near high-acuity bays ahead of "
            "EMS arrival, cutting defibrillation/epinephrine delays."
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
        dimensions="70 cm L × 46 cm W × 98 cm H",
        cost_estimate_usd="~$18k BOM target (slim chassis)",
        design_notes=(
            "Custom 46cm-wide narrow-track chassis with dual sealed specimen/med "
            "canisters designed specifically to clear legacy 0.72m–0.85m hospital doors."
        ),
        why_helpful=(
            "Unlocks autonomous delivery in older hospital wings where standard "
            "0.90m-clearance couriers are blocked."
        ),
    ),
    "amed_compounding_arm": RobotProfile(
        id="amed_compounding_arm",
        name="Amed 6-DOF Med-Prep Arm",
        vendor="Amed (Custom)",
        is_custom=True,
        speed_mps=1.20,
        clearance_m=0.85,
        length_m=0.80,
        compartments=2,
        handoff_seconds=8.0,
        supported_runs=["pharmacy_pickup", "lab_specimen_drop"],
        dimensions="80 cm reach · 6-DOF articulated arm on compact mobile pedestal",
        cost_estimate_usd="~$42k BOM target (6-DOF cobot arm + sterile gripper)",
        design_notes=(
            "6-axis articulated robotic arm with rotary base, dual harmonic-drive "
            "shoulder/elbow joints, sterile vial gripper, and optical barcode scanner "
            "mounted on a compact pedestal."
        ),
        why_helpful=(
            "Automates high-risk IV compounding, medication picking, and specimen "
            "rack transfers inside the Medication Room and Lab Drop."
        ),
        solves_problem="Medication & chemical handling delays",
        description=(
            "6-DOF articulated robotic arm for automated medication picking, "
            "IV prep, and specimen rack handoffs."
        ),
    ),
    "alta_transfer_robot": RobotProfile(
        id="alta_transfer_robot",
        name="ALTA Patient-Transfer Robot",
        vendor="ALTA (Lahey Reference)",
        is_custom=False,
        speed_mps=0.65,
        clearance_m=0.95,
        length_m=1.15,
        compartments=1,
        handoff_seconds=25.0,
        supported_runs=["supply_restock"],
        dimensions="115 cm L x 68 cm W x 135 cm H",
        cost_estimate_usd="~$85k-$120k unit",
        design_notes="Dual-actuator patient-lift and transfer mobile platform for corridor and room-entry safety testing.",
        why_helpful="Tests whether patient-transfer robotics can safely clear unit doorways and turn radii before deployment.",
        solves_problem="Patient handling and safe transfer",
        description="Patient-handling mobile platform for doorway clearance and corridor safety testing.",
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


MIXED_FLEET_ROLES: dict[str, str] = {
    "robot-1": "amed_compounding_arm",
    "robot-2": "amed_narrow_rover",
    "robot-3": "amed_prestage_pod",
}


def get_mixed_fleet_profile_id(agent_id: str) -> str:
    return MIXED_FLEET_ROLES.get(agent_id, "aethon_tug")
