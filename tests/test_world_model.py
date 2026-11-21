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
