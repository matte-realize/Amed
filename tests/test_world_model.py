"""Tests for World Model staff review/edit, EMS pre-staging, and regional coalition surge."""

from __future__ import annotations

from fastapi.testclient import TestClient

from hospital_twin.schemas import (
    CoalitionSurgeRequest,
    CoalitionUnitInput,
    DoorOverride,
    EmsArrivalSignal,
    RoomOverride,
)
from hospital_twin.world_model import (
    apply_staff_edits,
    evaluate_coalition_surge,
    plan_ems_prestaging,
)
from tests.test_simulation import _make_sample_unit


def test_apply_staff_edits_and_review_endpoint():
    unit = _make_sample_unit()
    edited = apply_staff_edits(
        unit,
        room_overrides=[RoomOverride(id="supply_room", autonomy_tier="human_only", label="Sterile Supply")],
        door_overrides=[DoorOverride(id="d2", width_m=0.82, robot_passable=False)],
    )
    # Original unit remains untouched
    assert next(r for r in unit.rooms if r.id == "supply_room").autonomy_tier != "human_only"
    assert next(r for r in edited.rooms if r.id == "supply_room").autonomy_tier == "human_only"
    assert next(r for r in edited.rooms if r.id == "supply_room").label == "Sterile Supply"
    assert next(d for d in edited.doors if d.id == "d2").width_m == 0.82
    assert next(d for d in edited.doors if d.id == "d2").robot_passable is False

    from hospital_twin.app import app

    client = TestClient(app)
    res = client.post(
        "/api/review-unit",
        json={
            "unit_map": unit.model_dump(),
            "room_overrides": [{"id": "med_room", "autonomy_tier": "autonomous"}],
            "door_overrides": [{"id": "d4", "width_m": 1.25, "robot_passable": True}],
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert next(r for r in body["rooms"] if r["id"] == "med_room")["autonomy_tier"] == "autonomous"
    assert next(d for d in body["doors"] if d["id"] == "d4")["width_m"] == 1.25


def test_ems_prestaging_and_endpoint():
    unit = _make_sample_unit()
    signal = EmsArrivalSignal(
        incident_id="EMS-2026-09",
        acuity="critical",
        eta_minutes=8.0,
        needed_runs=["pharmacy_pickup", "lab_specimen_drop"],
    )
    plan = plan_ems_prestaging(unit, signal)
    assert plan.incident_id == "EMS-2026-09"
    assert 0.0 < plan.pre_stage_minutes <= signal.eta_minutes
    assert len(plan.scheduled_runs) == 2

    from hospital_twin.app import app

    client = TestClient(app)
    res = client.post(
        "/api/ems-feed",
        json={"unit_map": unit.model_dump(), "signal": signal.model_dump()},
    )
    assert res.status_code == 200, res.text
    assert len(res.json()["scheduled_runs"]) == 2


def test_coalition_surge_and_endpoint():
    unit = _make_sample_unit()
    req = CoalitionSurgeRequest(
        hospitals=[
            CoalitionUnitInput(hospital_name="Metro General", unit_map=unit, num_robots=1),
            CoalitionUnitInput(hospital_name="Mercy East", unit_map=unit, num_robots=3),
        ],
        run_description="pharmacy pickup every 30 minutes, specimens hourly",
        shift_hours=8.0,
    )
    report = evaluate_coalition_surge(req)
    assert len(report.hospitals) == 2
    assert report.recommended_diversion == "Mercy East"

    from hospital_twin.app import app

    client = TestClient(app)
    res = client.post("/api/coalition-surge", json=req.model_dump())
    assert res.status_code == 200, res.text
    assert res.json()["recommended_diversion"] == "Mercy East"


def test_robotics_catalog_struggles_and_endpoints():
    from hospital_twin.robotics import ROBOT_CATALOG, HOSPITAL_STRUGGLES, get_robot_profile
    from hospital_twin.simulator import simulate_shift
    from hospital_twin.schemas import ScheduledRun

    assert len(ROBOT_CATALOG) == 7
    assert len(HOSPITAL_STRUGGLES) >= 4
    assert get_robot_profile("amed_narrow_rover").clearance_m == 0.72
    assert get_robot_profile("nonexistent").id == "aethon_tug"

    # Pneumatic tube handles pharmacy/lab but falls back to nurse on supply_restock
    unit = _make_sample_unit()
    schedule = [
        ScheduledRun(run_type="lab_specimen_drop", origin="nurse_station", destination="lab_drop", frequency_minutes=240.0),
        ScheduledRun(run_type="supply_restock", origin="supply_room", destination="nurse_station", frequency_minutes=240.0),
    ]
    comp = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, robot_profile_id="pneumatic_tube")
    assert comp.assumptions["robot_profile_id"] == "pneumatic_tube"
    assert any(e.agent_role == "robot" and e.run_type == "lab_specimen_drop" for e in comp.robot_assisted.events)
    assert any(e.agent_role == "nurse" and e.run_type == "supply_restock" for e in comp.robot_assisted.events)

    from hospital_twin.app import app

    client = TestClient(app)
    robots_res = client.get("/api/robots")
    assert robots_res.status_code == 200
    assert len(robots_res.json()["robots"]) == 7
    robots = robots_res.json()["robots"]
    assert len(robots) == 7
    for r in robots:
        for field in ("design_notes", "cost_estimate_usd", "why_helpful", "dimensions"):
            assert isinstance(r.get(field), str) and r[field].strip(), f"{r.get('id')} missing {field}"

    struggles_res = client.get("/api/struggles")
    assert struggles_res.status_code == 200
    assert len(struggles_res.json()["struggles"]) >= 4


def test_narrow_door_unlocked_by_slim_rover():
    from hospital_twin.schemas import ScheduledRun
    from hospital_twin.simulator import simulate_shift

    unit = _make_sample_unit()
    narrow_unit = apply_staff_edits(
        unit,
        room_overrides=[],
        door_overrides=[DoorOverride(id="d4", width_m=0.80, robot_passable=False)],
    )
    schedule = [
        ScheduledRun(
            run_type="supply_restock",
            origin="supply_room",
            destination="nurse_station",
            frequency_minutes=240.0,
        ),
    ]

    tug_comp = simulate_shift(
        narrow_unit,
        schedule,
        shift_hours=8.0,
        num_robots=1,
        robot_profile_id="aethon_tug",
    )
    tug_events = [e for e in tug_comp.robot_assisted.events if e.run_type == "supply_restock"]
    assert tug_events
    assert all(e.agent_role == "nurse" for e in tug_events)

    rover_comp = simulate_shift(
        narrow_unit,
        schedule,
        shift_hours=8.0,
        num_robots=1,
        robot_profile_id="amed_narrow_rover",
    )
    rover_events = [e for e in rover_comp.robot_assisted.events if e.run_type == "supply_restock"]
    assert rover_events
    assert all(e.agent_role == "robot" for e in rover_events)


def test_mixed_fleet_runs_distinct_custom_robots():
    from hospital_twin.schemas import ScheduledRun
    from hospital_twin.simulator import simulate_shift

    unit = _make_sample_unit()
    schedule = [
        ScheduledRun(run_type="pharmacy_pickup", origin="med_room", destination="nurse_station", frequency_minutes=30.0),
        ScheduledRun(run_type="lab_specimen_drop", origin="nurse_station", destination="lab_drop", frequency_minutes=30.0),
        ScheduledRun(run_type="supply_restock", origin="supply_room", destination="nurse_station", frequency_minutes=30.0),
    ]
    comp = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=3, robot_profile_id="mixed_fleet")
    assert comp.assumptions["robot_profile_id"] == "mixed_fleet"
    profiles_used = {e.robot_profile_id for e in comp.robot_assisted.events if e.agent_role == "robot"}
    assert "amed_compounding_arm" in profiles_used
    assert len(profiles_used) >= 2


def test_readiness_research_gap_and_additional_stress_modes():
    from hospital_twin.app import app
    from hospital_twin.schemas import ScheduledRun
    from hospital_twin.simulator import simulate_shift
    from hospital_twin.world_model import build_autonomy_plan, check_unit_readiness

    unit = _make_sample_unit()
    for d in unit.doors:
        if d.id == "d2":
            d.width_m = 0.80
            d.robot_passable = False

    plan = build_autonomy_plan(unit)
    assert any(rec.category == "research_gap" for rec in plan)

    report = check_unit_readiness(unit)
    assert report.total_items == 5
    assert len(report.flagged_items) >= 1

    client = TestClient(app)
    res = client.post("/api/readiness", json=unit.model_dump())
    assert res.status_code == 200
    assert res.json()["total_items"] == 5

    sched = [ScheduledRun(run_type="pharmacy_pickup", origin="med_room", destination="nurse_station", frequency_minutes=120.0)]
    ca = simulate_shift(unit, sched, shift_hours=8.0, num_robots=1, scenario_mode="cardiac_arrest", seed=3)
    cq = simulate_shift(unit, sched, shift_hours=8.0, num_robots=1, scenario_mode="contamination_quarantine", seed=3)
    assert ca.robot_assisted.total_trips == 4
    assert cq.robot_assisted.total_trips == 4


def test_catastrophic_mci_robots_outperform_overloaded_nurses():
    from hospital_twin.schemas import ScheduledRun
    from hospital_twin.simulator import simulate_shift

    unit = _make_sample_unit()
    schedule = [
        ScheduledRun(run_type="pharmacy_pickup", origin="med_room", destination="nurse_station", frequency_minutes=30.0),
        ScheduledRun(run_type="lab_specimen_drop", origin="nurse_station", destination="lab_drop", frequency_minutes=30.0),
        ScheduledRun(run_type="supply_restock", origin="supply_room", destination="nurse_station", frequency_minutes=30.0),
    ]
    comp = simulate_shift(
        unit,
        schedule,
        shift_hours=8.0,
        num_robots=3,
        scenario_mode="catastrophic_mci",
        seed=42,
        robot_profile_id="mixed_fleet",
    )
    assert comp.assumptions["scenario_mode"] == "catastrophic_mci"
    assert comp.assumptions["stat_trips"] == 8
    assert comp.nurse_minutes_saved > 20
    assert comp.robot_assisted.avg_delivery_wait_s < comp.nurse_only.avg_delivery_wait_s
