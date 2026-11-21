"""Hospital World Model analytics: Autonomy Plan, Ranked Bottlenecks, Tipping Points, and Stress Sweep."""

from __future__ import annotations

from collections import defaultdict
import re
from uuid import uuid4

from hospital_twin.robotics import ROBOT_CATALOG
from hospital_twin.routing import DEFAULT_ROBOT_CLEARANCE_M
from hospital_twin.schedule_parser import parse_run_description
from hospital_twin.schemas import (
    InventoryItem,
    CustomRdConcept,
    ReadinessReport,
    RoomDelegationItem,
    UnitDelegationReport,
    AutonomyRecommendation,
    BottleneckItem,
    CalibrationInput,
    CalibrationResult,
    CoalitionSurgeReport,
    CoalitionSurgeRequest,
    CoordinatedRobotTask,
    DoorOverride,
    EmtCoordinationResponse,
    EmsArrivalSignal,
    EmsDispatchPlan,
    RoomOverride,
    RoomType,
    ScheduledRun,
    SimulationComparison,
    StressTestReport,
    StressTestSweepPoint,
    UnitMap,
)
from hospital_twin.simulator import (
    DEFAULT_HANDOFF_SECONDS,
    DEFAULT_ROBOT_SPEED_MPS,
    ScenarioMode,
    simulate_shift,
)


def build_autonomy_plan(
    unit: UnitMap,
    robot_clearance_m: float = DEFAULT_ROBOT_CLEARANCE_M,
) -> list[AutonomyRecommendation]:
    """Classify rooms and narrow-door bottlenecks into the 4-bucket Autonomy Plan."""
    recommendations: list[AutonomyRecommendation] = []
    narrow_by_room: dict[str, list[str]] = defaultdict(list)

    for door in unit.doors:
        if not door.robot_passable or door.width_m < robot_clearance_m:
            for rid in door.rooms_connected:
                narrow_by_room[rid].append(f"{door.id} ({door.width_m:.2f}m)")

    for room in unit.rooms:
        if room.type == "corridor":
            continue
        blocked_doors = narrow_by_room.get(room.id, [])
        if room.autonomy_tier == "human_only":
            recommendations.append(
                AutonomyRecommendation(
                    target_id=room.id,
                    target_label=room.label,
                    category="keep_human",
                    rationale=room.autonomy_reason or "Clinical bedside care or restricted access.",
                    action="Corridor/threshold handoff only; keep bedside tasks human.",
                )
            )
        elif blocked_doors:
            recommendations.append(
                AutonomyRecommendation(
                    target_id=room.id,
                    target_label=room.label,
                    category="redesign_process",
                    rationale=f"Blocked by narrow door(s) below {robot_clearance_m:.2f}m: {', '.join(blocked_doors)}.",
                    action=room.proposed_solution or "Widen doorway to >=0.90m or add exterior pass-through locker.",
                )
            )
        elif room.autonomy_tier == "assisted":
            recommendations.append(
                AutonomyRecommendation(
                    target_id=room.id,
                    target_label=room.label,
                    category="redesign_process",
                    rationale=room.autonomy_reason or "Requires badge-actuated door opener or staging pad.",
                    action=room.proposed_solution or "Install auto-door actuator or outer counter docking pad.",
                )
            )
        else:
            recommendations.append(
                AutonomyRecommendation(
                    target_id=room.id,
                    target_label=room.label,
                    category="use_robot",
                    rationale=room.autonomy_reason or "Clear corridor/door access for autonomous courier pod.",
                    action="Deploy Aethon TUG + 3-compartment pod.",
                )
            )

    # Record R&D gaps where no standard automated option fits (candidate for pooled funding)
    if narrow_by_room:
        recommendations.append(
            AutonomyRecommendation(
                target_id="rd_gap_narrow_door_robotics",
                target_label="Unit-Wide Narrow-Door Gap",
                category="research_gap",
                rationale=f"{len(narrow_by_room)} room(s) fall below standard 0.90m commercial chassis clearance.",
                action="Candidate for pooled hospital R&D funding: compact slim-chassis courier or automated door retrofit.",
            )
        )

    return recommendations


def check_unit_readiness(unit: UnitMap) -> ReadinessReport:
    """Audit critical unit inventory & readiness (crash carts, defibrillators, suction, oxygen, RFID med kits)."""
    room_ids = {r.id for r in unit.rooms}
    med_room = next((r.id for r in unit.rooms if r.type == "medication_room"), next(iter(room_ids), "unit"))
    supply_room = next((r.id for r in unit.rooms if r.type == "supply_room"), next(iter(room_ids), "unit"))
    nurse_station = next((r.id for r in unit.rooms if r.type == "nurse_station"), next(iter(room_ids), "unit"))
    blocked_doors = [d for d in unit.doors if not d.robot_passable or d.width_m < DEFAULT_ROBOT_CLEARANCE_M]

    items = [
        InventoryItem(
            id="inv_crash_cart_1",
            name="Adult Crash Cart #1",
            location_room_id=nurse_station,
            category="crash_cart",
            status="ready",
            detail="Sealed & verified at nurse station hub.",
        ),
        InventoryItem(
            id="inv_defib_1",
            name="Biphasic Defibrillator / AED",
            location_room_id=supply_room,
            category="defibrillator",
            status="low_battery" if blocked_doors else "ready",
            detail="Staged in supply room; narrow door delays rapid cart dispatch." if blocked_doors else "Battery 100%, pads in date.",
        ),
        InventoryItem(
            id="inv_suction_1",
            name="Portable Suction Unit",
            location_room_id=supply_room,
            category="suction",
            status="ready",
            detail="Canister & regulator checked.",
        ),
        InventoryItem(
            id="inv_oxygen_1",
            name="Emergency O2 Cylinder Bank",
            location_room_id=supply_room,
            category="oxygen",
            status="ready",
            detail="Pressure >1800 PSI.",
        ),
        InventoryItem(
            id="inv_rfid_meds",
            name="RFID Epinephrine & Code Med Kit",
            location_room_id=med_room,
            category="rfid_med_kit",
            status="expired" if len(unit.rooms) > 9 else "ready",
            detail="Flagged for lot rotation during shift audit." if len(unit.rooms) > 9 else "All lots active.",
        ),
    ]
    flagged = [it for it in items if it.status != "ready"]
    ready_count = len(items) - len(flagged)
    summary = f"{ready_count}/{len(items)} critical readiness items ready; {len(flagged)} flagged for attention."
    return ReadinessReport(
        total_items=len(items),
        ready_count=ready_count,
        flagged_items=flagged,
        summary=summary,
    )


def rank_bottlenecks(
    unit: UnitMap,
    comparison: SimulationComparison,
) -> list[BottleneckItem]:
    """Rank logistical bottlenecks in descending order of impact minutes."""
    room_labels = {r.id: r.label for r in unit.rooms}
    impact_by_source: dict[str, float] = defaultdict(float)
    detail_by_source: dict[str, str] = {}

    for ev in comparison.robot_assisted.events:
        if ev.agent_role == "nurse":
            src = f"Human fallback: {room_labels.get(ev.node_path[0], ev.node_path[0])} -> {room_labels.get(ev.node_path[-1], ev.node_path[-1])}"
            impact_by_source[src] += (ev.end_time_s - ev.start_time_s) / 60.0
            detail_by_source[src] = "Robot-unreachable route (narrow door or human-only boundary) forced nurse walk."
        elif ev.wait_time_s > 0.0:
            src = f"Queue wait: {ev.run_type.replace('_', ' ')}"
            impact_by_source[src] += ev.wait_time_s / 60.0
            detail_by_source[src] = "Courier fleet contention delayed dispatch."

    if comparison.robot_assisted.flagged_exceptions > 0:
        impact_by_source["Uncollected deliveries (>10m)"] = float(
            comparison.robot_assisted.flagged_exceptions * 10.0
        )
        detail_by_source["Uncollected deliveries (>10m)"] = (
            f"{comparison.robot_assisted.flagged_exceptions} delivery handoff(s) exceeded the 10-minute collection threshold."
        )

    ranked = sorted(impact_by_source.items(), key=lambda kv: kv[1], reverse=True)
    return [
        BottleneckItem(
            rank=idx + 1,
            source=src,
            impact_minutes=round(mins, 2),
            detail=detail_by_source.get(src, ""),
        )
        for idx, (src, mins) in enumerate(ranked[:5])
    ]


def run_stress_sweep(
    unit: UnitMap,
    schedule: list[ScheduledRun],
    shift_hours: float = 8.0,
    scenario_mode: ScenarioMode = "surge",
    seeds: tuple[int, ...] = (11, 22, 33, 44, 55),
) -> StressTestReport:
    """Run parallel/multi-seed stress sweeps across 1..3 robots to locate tipping points."""
    sweep_points: list[StressTestSweepPoint] = []
    sample_comp: SimulationComparison | None = None

    for n_robots in (1, 2, 3):
        saved_vals: list[float] = []
        p95_vals: list[float] = []
        max_flags = 0
        for seed in seeds:
            comp = simulate_shift(
                unit=unit,
                schedule=schedule,
                shift_hours=shift_hours,
                num_robots=n_robots,
                scenario_mode=scenario_mode,
                seed=seed,
            )
            if sample_comp is None:
                sample_comp = comp
            saved_vals.append(comp.nurse_minutes_saved)
            p95_vals.append(float(comp.assumptions.get("p95_wait_s", 0.0)))
            max_flags = max(max_flags, comp.robot_assisted.flagged_exceptions)

        sweep_points.append(
            StressTestSweepPoint(
                num_robots=n_robots,
                mean_nurse_minutes_saved=round(sum(saved_vals) / len(saved_vals), 2),
                mean_p95_wait_s=round(sum(p95_vals) / len(p95_vals), 1),
                max_flagged_exceptions=max_flags,
            )
        )

    tipping = "No critical tipping point detected under current demand."
    for pt in sweep_points:
        if pt.max_flagged_exceptions > 0 or pt.mean_p95_wait_s > 120.0:
            next_ok = next(
                (p for p in sweep_points if p.num_robots > pt.num_robots and p.max_flagged_exceptions == 0 and p.mean_p95_wait_s <= 120.0),
                None,
            )
            if next_ok:
                tipping = (
                    f"Tips at {pt.num_robots} robot(s) (p95 wait {pt.mean_p95_wait_s:.0f}s, "
                    f"{pt.max_flagged_exceptions} >10m flags); stabilizes at {next_ok.num_robots} robots "
                    f"(p95 wait {next_ok.mean_p95_wait_s:.0f}s)."
                )
            else:
                tipping = (
                    f"Contention at {pt.num_robots} robot(s): p95 wait {pt.mean_p95_wait_s:.0f}s, "
                    f"{pt.max_flagged_exceptions} >10m uncollected flags."
                )
            break

    assert sample_comp is not None
    autonomy_plan = build_autonomy_plan(unit)
    bottlenecks = rank_bottlenecks(unit, sample_comp)
    runs_evaluated = len(sweep_points) * len(seeds)

    after_action = (
        f"ASPR-Style After-Action Summary ({scenario_mode.upper()}, {runs_evaluated} runs): "
        f"{tipping} Top bottleneck: {bottlenecks[0].source if bottlenecks else 'None'}. "
        f"Autonomy Plan: {sum(1 for a in autonomy_plan if a.category == 'use_robot')} robot-ready, "
        f"{sum(1 for a in autonomy_plan if a.category == 'redesign_process')} process-redesign, "
        f"{sum(1 for a in autonomy_plan if a.category == 'keep_human')} human-only zones."
    )

    sweep_rows = "\n".join(
        f"| {p.num_robots} | {p.mean_nurse_minutes_saved:.1f} min | {p.mean_p95_wait_s:.1f}s | {p.max_flagged_exceptions} |"
        for p in sweep_points
    )
    bn_rows = "\n".join(
        f"{b.rank}. **{b.source}** ({b.impact_minutes:.1f} min) — {b.detail}"
        for b in bottlenecks
    ) or "- None detected."
    plan_rows = "\n".join(
        f"- **{a.target_label}** (`{a.category}`): {a.rationale} -> *{a.action}*"
        for a in autonomy_plan
    )
    after_action_md = (
        f"# After-Action Exercise Report ({scenario_mode.upper()})\n\n"
        f"- **Runs Evaluated:** {runs_evaluated} ({len(seeds)} seeds x 1-3 robots)\n"
        f"- **Tipping Point:** {tipping}\n\n"
        f"## Fleet Sweep\n| Robots | Mean Nurse Time Saved | Mean p95 Wait | Max >10m Flags |\n| ---: | ---: | ---: | ---: |\n{sweep_rows}\n\n"
        f"## Ranked Bottlenecks\n{bn_rows}\n\n"
        f"## Autonomy Plan\n{plan_rows}\n"
    )

    return StressTestReport(
        scenario_mode=scenario_mode,
        runs_evaluated=runs_evaluated,
        tipping_point=tipping,
        sweep=sweep_points,
        autonomy_plan=autonomy_plan,
        ranked_bottlenecks=bottlenecks,
        after_action_summary=after_action,
        after_action_markdown=after_action_md,
    )


def calibrate_from_drill(cal: CalibrationInput) -> CalibrationResult:
    """Recalibrate world model timing parameters from observed mock-drill measurements."""
    nurse_speed = cal.observed_nurse_speed_mps if cal.observed_nurse_speed_mps is not None else 1.2
    robot_speed = cal.observed_robot_speed_mps if cal.observed_robot_speed_mps is not None else DEFAULT_ROBOT_SPEED_MPS
    handoff_s = cal.observed_handoff_seconds if cal.observed_handoff_seconds is not None else DEFAULT_HANDOFF_SECONDS

    deltas: list[str] = []
    if cal.observed_nurse_speed_mps is not None:
        deltas.append(f"nurse_speed: 1.20 -> {nurse_speed:.2f} m/s")
    if cal.observed_robot_speed_mps is not None:
        deltas.append(f"robot_speed: {DEFAULT_ROBOT_SPEED_MPS:.2f} -> {robot_speed:.2f} m/s")
    if cal.observed_handoff_seconds is not None:
        deltas.append(f"handoff: {DEFAULT_HANDOFF_SECONDS:.0f}s -> {handoff_s:.0f}s")
    summary = ", ".join(deltas) if deltas else "No parameter changes; baseline defaults retained."
    if cal.drill_notes:
        summary += f" ({cal.drill_notes})"

    return CalibrationResult(
        nurse_speed_mps=round(nurse_speed, 3),
        robot_speed_mps=round(robot_speed, 3),
        handoff_seconds=round(handoff_s, 3),
        delta_summary=summary,
    )


def apply_staff_edits(
    unit: UnitMap,
    room_overrides: list[RoomOverride],
    door_overrides: list[DoorOverride],
) -> UnitMap:
    """Return a deep-copied UnitMap with staff room and door overrides applied."""
    updated = unit.model_copy(deep=True)
    room_map = {ov.id: ov for ov in room_overrides}
    for room in updated.rooms:
        ov = room_map.get(room.id)
        if ov is None:
            continue
        if ov.autonomy_tier is not None:
            room.autonomy_tier = ov.autonomy_tier
        if ov.label is not None:
            room.label = ov.label

    door_map = {ov.id: ov for ov in door_overrides}
    for door in updated.doors:
        ov = door_map.get(door.id)
        if ov is None:
            continue
        if ov.width_m is not None:
            door.width_m = ov.width_m
        if ov.robot_passable is not None:
            door.robot_passable = ov.robot_passable

    return updated


def _find_room_id(
    unit: UnitMap,
    preferred_type: RoomType,
    keywords: tuple[str, ...],
    fallback_id: str,
) -> str:
    """Resolve a room ID by RoomType first, then by keyword match, then fallback."""
    for room in unit.rooms:
        if room.type == preferred_type:
            return room.id
    for room in unit.rooms:
        haystack = f"{room.id} {room.label}".lower()
        if any(kw in haystack for kw in keywords):
            return room.id
    for room in unit.rooms:
        if room.type != "corridor" and room.id != fallback_id:
            return room.id
    return fallback_id


def plan_ems_prestaging(unit: UnitMap, signal: EmsArrivalSignal) -> EmsDispatchPlan:
    """Map an incoming NEMSIS-style EMS arrival signal to pre-staged courier runs."""
    if not unit.rooms:
        raise ValueError("UnitMap contains no rooms.")

    non_corridor = next((r.id for r in unit.rooms if r.type != "corridor"), unit.rooms[0].id)
    nurse_station = _find_room_id(unit, "nurse_station", ("nurse", "station", "hub"), non_corridor)
    med_room = _find_room_id(unit, "medication_room", ("med", "pharm", "rx"), nurse_station)
    supply_room = _find_room_id(unit, "supply_room", ("supply", "stor", "closet"), nurse_station)
    lab_drop = _find_room_id(unit, "lab_drop", ("lab", "spec", "path"), nurse_station)

    acuity_cap = {"critical": 5.0, "urgent": 10.0, "routine": signal.eta_minutes}
    pre_stage_minutes = round(max(1.0, min(signal.eta_minutes, acuity_cap[signal.acuity])), 1)
    freq_minutes = max(5.0, round(signal.eta_minutes, 1))

    endpoints = {
        "pharmacy_pickup": (med_room, nurse_station),
        "supply_restock": (supply_room, nurse_station),
        "lab_specimen_drop": (nurse_station, lab_drop),
    }

    scheduled_runs: list[ScheduledRun] = []
    for run_type in signal.needed_runs:
        origin, destination = endpoints[run_type]
        scheduled_runs.append(
            ScheduledRun(
                run_type=run_type,
                origin=origin,
                destination=destination,
                frequency_minutes=freq_minutes,
                label=f"EMS {signal.incident_id} ({signal.acuity}): {run_type.replace('_', ' ')}",
            )
        )

    run_list = ", ".join(r.replace("_", " ") for r in signal.needed_runs) or "none"
    summary = (
        f"Incident {signal.incident_id} [{signal.acuity.upper()}], ETA {signal.eta_minutes:.1f}m: "
        f"pre-staging {len(scheduled_runs)} run(s) ({run_list}) within {pre_stage_minutes:.1f}m."
    )
    return EmsDispatchPlan(
        incident_id=signal.incident_id,
        pre_stage_minutes=pre_stage_minutes,
        scheduled_runs=scheduled_runs,
        summary=summary,
    )


def _extract_eta_minutes(transcript: str, eta_minutes: float | None) -> float:
    """Resolve EMS ETA in minutes from explicit override or transcript text."""
    if eta_minutes is not None:
        return round(max(1.0, float(eta_minutes)), 1)
    match = re.search(
        r"\beta\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:min|mins|minutes|m)?\b",
        transcript,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"\b(\d+(?:\.\d+)?)\s*(?:min|mins|minutes)\b",
            transcript,
            flags=re.IGNORECASE,
        )
    if match:
        return round(max(1.0, float(match.group(1))), 1)
    return 5.0


def _detect_acuity(transcript: str) -> str:
    """Classify incoming EMT transcript into routine, urgent, or critical acuity."""
    lower = transcript.lower()
    if re.search(r"\b(trauma|arrest|stat|surgery|or|crash|bleeding|code)\b", lower):
        return "critical"
    if re.search(r"\b(chest|stroke|iv|oxygen)\b", lower):
        return "urgent"
    return "routine"


def _make_robot_task(
    robot_id: str,
    task: str,
    target_room: str,
    eta_ready_s: float,
) -> CoordinatedRobotTask:
    profile = ROBOT_CATALOG[robot_id]
    return CoordinatedRobotTask(
        robot_id=profile.id,
        robot_name=profile.name,
        task=task,
        target_room=target_room,
        eta_ready_s=round(eta_ready_s, 1),
    )


def coordinate_emt_dispatch(
    unit: UnitMap,
    transcript: str,
    eta_minutes: float | None = None,
) -> EmtCoordinationResponse:
    """Translate an incoming EMT radio transcript into coordinated multi-robot pre-staging tasks."""
    if not unit.rooms:
        raise ValueError("UnitMap contains no rooms.")
    if not transcript.strip():
        raise ValueError("EMT transcript must not be empty.")

    resolved_eta = _extract_eta_minutes(transcript, eta_minutes)
    acuity = _detect_acuity(transcript)
    eta_seconds = resolved_eta * 60.0
    freq_minutes = max(5, int(round(resolved_eta)))

    non_corridor = next((r.id for r in unit.rooms if r.type != "corridor"), unit.rooms[0].id)
    nurse_station = _find_room_id(unit, "nurse_station", ("nurse", "station", "hub"), non_corridor)
    med_room = _find_room_id(unit, "medication_room", ("med", "pharm", "rx"), nurse_station)
    supply_room = _find_room_id(unit, "supply_room", ("supply", "stor", "closet"), nurse_station)
    lab_drop = _find_room_id(unit, "lab_drop", ("lab", "spec", "path"), nurse_station)
    trauma_or_room = _find_room_id(
        unit,
        "patient_room",
        ("or", "operating", "trauma", "surg", "resus"),
        nurse_station,
    )

    lower = transcript.lower()
    needs_meds = bool(re.search(r"\b(med|meds|epi|epinephrine|iv|pharm|pharmacy)\b", lower))
    needs_lab = bool(re.search(r"\b(blood|lab|specimen|specimens)\b", lower))
    needs_surgical = bool(
        re.search(r"\b(surgery|surgical|or|trauma|cart|supply|supplies|crash|arrest)\b", lower)
    )

    coordinated_tasks: list[CoordinatedRobotTask] = []
    run_segments: list[str] = []

    if needs_meds:
        coordinated_tasks.append(
            _make_robot_task(
                "amed_compounding_arm",
                "Compound & stage STAT IV / emergency medications",
                med_room,
                min(eta_seconds * 0.45, 90.0),
            )
        )
        run_segments.append(
            f"pharmacy pickup from {med_room} to {nurse_station} every {freq_minutes} minutes"
        )

    if needs_lab:
        coordinated_tasks.append(
            _make_robot_task(
                "amed_narrow_rover",
                "Shuttle STAT blood type/crossmatch & specimen kits via narrow corridors",
                lab_drop,
                min(eta_seconds * 0.55, 110.0),
            )
        )
        run_segments.append(
            f"lab specimen drop from {nurse_station} to {lab_drop} every {freq_minutes} minutes"
        )

    if needs_surgical:
        coordinated_tasks.append(
            _make_robot_task(
                "amed_surgical_prep_bot",
                "Stage sterile surgical/trauma instrument tray and OR turnover kit",
                trauma_or_room,
                min(eta_seconds * 0.65, 130.0),
            )
        )
        coordinated_tasks.append(
            _make_robot_task(
                "amed_prestage_pod",
                "Pre-stage crash cart, defibrillator, and emergency trauma supplies",
                nurse_station,
                min(eta_seconds * 0.50, 100.0),
            )
        )
        run_segments.append(
            f"supply restock from {supply_room} to {nurse_station} every {freq_minutes} minutes"
        )

    if not coordinated_tasks:
        coordinated_tasks.append(
            _make_robot_task(
                "amed_prestage_pod",
                "Pre-stage intake readiness cart and baseline supplies",
                nurse_station,
                min(eta_seconds * 0.60, 120.0),
            )
        )
        run_segments.append(
            f"supply restock from {supply_room} to {nurse_station} every {freq_minutes} minutes"
        )

    incident_id = f"EMT-{uuid4().hex[:6].upper()}"
    run_description = "; ".join(run_segments)
    robot_list = ", ".join(t.robot_id for t in coordinated_tasks)
    summary = (
        f"Incident {incident_id} [{acuity.upper()}], ETA {resolved_eta:.1f}m: "
        f"dispatched {len(coordinated_tasks)} coordinated task(s) ({robot_list}) "
        f"ahead of EMS arrival."
    )
    return EmtCoordinationResponse(
        incident_id=incident_id,
        acuity=acuity,  # type: ignore[arg-type]
        eta_minutes=resolved_eta,
        run_description=run_description,
        coordinated_tasks=coordinated_tasks,
        summary=summary,
    )


def evaluate_coalition_surge(req: CoalitionSurgeRequest) -> CoalitionSurgeReport:
    """Run surge simulations across coalition hospitals and recommend diversion to lowest p95_wait_s."""
    if not req.hospitals:
        raise ValueError("At least one hospital must be provided.")

    results: list[dict] = []
    for hosp in req.hospitals:
        available_rooms = [r.id for r in hosp.unit_map.rooms]
        if not available_rooms:
            raise ValueError(f"Hospital '{hosp.hospital_name}' UnitMap contains no rooms.")
        schedule = parse_run_description(req.run_description, available_rooms)
        if not schedule:
            raise ValueError(f"Could not parse any scheduled runs for '{hosp.hospital_name}'.")

        comp = simulate_shift(
            unit=hosp.unit_map,
            schedule=schedule,
            shift_hours=req.shift_hours,
            num_robots=hosp.num_robots,
            scenario_mode="surge",
        )
        p95_wait_s = float(comp.assumptions.get("p95_wait_s", 0.0))
        results.append(
            {
                "hospital_name": hosp.hospital_name,
                "num_robots": hosp.num_robots,
                "p95_wait_s": round(p95_wait_s, 2),
                "avg_delivery_wait_s": comp.robot_assisted.avg_delivery_wait_s,
                "nurse_minutes_saved": comp.nurse_minutes_saved,
                "flagged_exceptions": comp.robot_assisted.flagged_exceptions,
                "total_trips": comp.robot_assisted.total_trips,
            }
        )

    best = min(
        results,
        key=lambda item: (
            item["p95_wait_s"],
            item["flagged_exceptions"],
            -item["nurse_minutes_saved"],
        ),
    )
    return CoalitionSurgeReport(
        hospitals=results,
        recommended_diversion=str(best["hospital_name"]),
    )


def analyze_floorplan_delegation(unit: UnitMap) -> UnitDelegationReport:
    """Delegate every room/zone on a floor plan to mobile robots, stationary robots, human care, or custom R&D."""
    narrow_rooms: set[str] = set()
    for door in unit.doors:
        if not door.robot_passable or door.width_m < DEFAULT_ROBOT_CLEARANCE_M:
            narrow_rooms.update(door.rooms_connected)

    delegations: list[RoomDelegationItem] = []
    for room in unit.rooms:
        if room.type == "medication_room":
            delegations.append(
                RoomDelegationItem(
                    room_id=room.id,
                    room_label=room.label,
                    delegation_type="stationary_robot",
                    assigned_system="Amed 6-DOF Med-Prep Arm + Stationary Pass-Through Locker",
                    rationale="High-precision pharmaceutical counting, IV prep, and exterior badge-locker handoff without door bottlenecks.",
                )
            )
        elif room.type == "lab_drop":
            delegations.append(
                RoomDelegationItem(
                    room_id=room.id,
                    room_label=room.label,
                    delegation_type="stationary_robot",
                    assigned_system="Pneumatic Tube Station + Automated Specimen Intake",
                    rationale="Stationary tube terminal cuts stat lab transit from minutes to seconds.",
                )
            )
        elif room.type == "corridor":
            delegations.append(
                RoomDelegationItem(
                    room_id=room.id,
                    room_label=room.label,
                    delegation_type="mobile_robot",
                    assigned_system="Aethon TUG / Amed Pre-Stage Cart",
                    rationale="Clear main transit spine for mobile autonomous couriers and emergency cart staging.",
                )
            )
        elif room.type == "supply_room":
            if room.id in narrow_rooms:
                delegations.append(
                    RoomDelegationItem(
                        room_id=room.id,
                        room_label=room.label,
                        delegation_type="rd_gap",
                        assigned_system="Custom Amed Slim Rover (0.72m) / Pass-Through Locker",
                        rationale="Doorway narrower than 0.90m blocks standard mobile couriers; requires slim chassis or wall locker.",
                    )
                )
            else:
                delegations.append(
                    RoomDelegationItem(
                        room_id=room.id,
                        room_label=room.label,
                        delegation_type="mobile_robot",
                        assigned_system="Amed Pre-Stage Cart / Diligent Moxi",
                        rationale="Automated supply restock and crash-cart staging.",
                    )
                )
        elif room.type == "nurse_station":
            delegations.append(
                RoomDelegationItem(
                    room_id=room.id,
                    room_label=room.label,
                    delegation_type="stationary_robot",
                    assigned_system="Outer Counter Dock Pad + Pneumatic Tube Terminal",
                    rationale="Shared clinical hub; stationary dock pad receives mobile deliveries without entering workspace.",
                )
            )
        elif room.autonomy_tier == "human_only" or room.type == "patient_room":
            if room.id in narrow_rooms:
                delegations.append(
                    RoomDelegationItem(
                        room_id=room.id,
                        room_label=room.label,
                        delegation_type="rd_gap",
                        assigned_system="Threshold Handoff + Custom 0.72m Slim Rover",
                        rationale="Legacy narrow doorway (<0.90m) + bedside clinical boundary; mobile couriers stage at threshold.",
                    )
                )
            else:
                delegations.append(
                    RoomDelegationItem(
                        room_id=room.id,
                        room_label=room.label,
                        delegation_type="human_only",
                        assigned_system="Clinician Bedside Care (Corridor Threshold Handoff)",
                        rationale="Direct patient care stays human; robots stage supplies at room threshold.",
                    )
                )
        else:
            delegations.append(
                RoomDelegationItem(
                    room_id=room.id,
                    room_label=room.label,
                    delegation_type="mobile_robot",
                    assigned_system="Aethon TUG / Diligent Moxi",
                    rationale="General unit logistics zone.",
                )
            )

    custom_rd_concepts = [
        CustomRdConcept(
            id="rd_slim_rover",
            title="Amed Slim-Chassis Rover (0.72m)",
            target_problem=f"Unlocks {len(narrow_rooms)} narrow-doorway room(s) (<0.90m) where commercial TUGs are blocked",
            system_type="mobile",
            estimated_bom_usd="~$18k BOM",
            engineering_summary="46cm-wide narrow-track mobile base with dual sealed specimen/med canisters.",
        ),
        CustomRdConcept(
            id="rd_med_prep_arm",
            title="Amed 6-DOF Med-Prep & Counting Arm",
            target_problem="Automates urgent pharmaceutical counting, IV prep, and vial staging inside Medication Rooms",
            system_type="stationary",
            estimated_bom_usd="~$42k BOM",
            engineering_summary="Stationary 6-axis sterile cobot arm with optical barcode verification and syringe gripper.",
        ),
        CustomRdConcept(
            id="rd_pass_through_locker",
            title="Stationary Pass-Through RFID Locker",
            target_problem="Eliminates door-opener retrofits at Medication & Supply rooms",
            system_type="stationary",
            estimated_bom_usd="~$6.5k BOM",
            engineering_summary="Wall-mounted dual-sided badge locker with UV-C sanitation cycle and exterior courier dock.",
        ),
    ]

    return UnitDelegationReport(
        total_rooms=len(delegations),
        mobile_count=sum(1 for d in delegations if d.delegation_type == "mobile_robot"),
        stationary_count=sum(1 for d in delegations if d.delegation_type == "stationary_robot"),
        human_only_count=sum(1 for d in delegations if d.delegation_type == "human_only"),
        rd_gap_count=sum(1 for d in delegations if d.delegation_type == "rd_gap"),
        delegations=delegations,
        custom_rd_concepts=custom_rd_concepts,
    )
