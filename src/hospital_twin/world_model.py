"""Hospital World Model analytics: Autonomy Plan, Ranked Bottlenecks, Tipping Points, and Stress Sweep."""

from __future__ import annotations

from collections import defaultdict

from hospital_twin.routing import DEFAULT_ROBOT_CLEARANCE_M
from hospital_twin.schemas import (
    AutonomyRecommendation,
    BottleneckItem,
    ScheduledRun,
    SimulationComparison,
    StressTestReport,
    StressTestSweepPoint,
    UnitMap,
)
from hospital_twin.simulator import ScenarioMode, simulate_shift


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

    return recommendations


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

    return StressTestReport(
        scenario_mode=scenario_mode,
        runs_evaluated=runs_evaluated,
        tipping_point=tipping,
        sweep=sweep_points,
        autonomy_plan=autonomy_plan,
        ranked_bottlenecks=bottlenecks,
        after_action_summary=after_action,
    )
