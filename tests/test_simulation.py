"""Tests for route graph, Hydrogen schedule parser, and SimPy shift simulation."""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hospital_twin.routing import (
    build_route_graph,
    find_robot_handoff_node,
    room_centroid,
    shortest_path_with_pixels,
)
from hospital_twin.schedule_parser import parse_run_description
from hospital_twin.schemas import Door, Point, Room, ScheduledRun, UnitMap
from hospital_twin.simulator import simulate_shift


def _make_sample_unit(meters_per_pixel: float = 0.05) -> UnitMap:
    """Build a 5-room hospital unit connected through a central corridor.

    Layout (pixels):
      - corridor:      centroid (300, 200)
      - nurse_station: centroid (300, 100), door at (300, 150) -> dist 100 px = 5 m
      - med_room:      centroid (100, 200), door at (200, 200) -> dist 200 px = 10 m
      - lab_drop:      centroid (500, 200), door at (400, 200) -> dist 200 px = 10 m
      - supply_room:   centroid (300, 350), door at (300, 250) -> dist 150 px = 7.5 m
    """
    rooms = [
        Room(
            id="corridor",
            label="Main Corridor",
            type="corridor",
            polygon=[
                Point(x=200, y=150),
                Point(x=400, y=150),
                Point(x=400, y=250),
                Point(x=200, y=250),
            ],
        ),
        Room(
            id="nurse_station",
            label="Nurse Station",
            type="nurse_station",
            polygon=[
                Point(x=250, y=50),
                Point(x=350, y=50),
                Point(x=350, y=150),
                Point(x=250, y=150),
            ],
            centroid=Point(x=300, y=100),
        ),
        Room(
            id="med_room",
            label="Medication Room",
            type="medication_room",
            polygon=[
                Point(x=0, y=150),
                Point(x=200, y=150),
                Point(x=200, y=250),
                Point(x=0, y=250),
            ],
            centroid=Point(x=100, y=200),
        ),
        Room(
            id="lab_drop",
            label="Lab Drop",
            type="lab_drop",
            polygon=[
                Point(x=400, y=150),
                Point(x=600, y=150),
                Point(x=600, y=250),
                Point(x=400, y=250),
            ],
            centroid=Point(x=500, y=200),
        ),
        Room(
            id="supply_room",
            label="Supply Room",
            type="supply_room",
            polygon=[
                Point(x=250, y=250),
                Point(x=350, y=250),
                Point(x=350, y=450),
                Point(x=250, y=450),
            ],
            centroid=Point(x=300, y=350),
        ),
    ]
    doors = [
        Door(id="d1", position=Point(x=300, y=150), rooms_connected=["nurse_station", "corridor"]),
        Door(id="d2", position=Point(x=200, y=200), rooms_connected=["med_room", "corridor"]),
        Door(id="d3", position=Point(x=400, y=200), rooms_connected=["lab_drop", "corridor"]),
        Door(id="d4", position=Point(x=300, y=250), rooms_connected=["supply_room", "corridor"]),
    ]
    return UnitMap(
        image_width=600,
        image_height=500,
        meters_per_pixel=meters_per_pixel,
        rooms=rooms,
        doors=doors,
    )


def test_graph_connectivity_and_centroid():
    unit = _make_sample_unit()
    corridor = unit.rooms[0]
    assert corridor.centroid is None
    c = room_centroid(corridor)
    assert math.isclose(c.x, 300.0, rel_tol=1e-6)
    assert math.isclose(c.y, 200.0, rel_tol=1e-6)

    graph = build_route_graph(unit)
    assert set(graph.nodes) == {"corridor", "nurse_station", "med_room", "lab_drop", "supply_room"}
    assert graph.number_of_edges() == 4
    for u in ("nurse_station", "med_room", "lab_drop", "supply_room"):
        assert graph.has_edge(u, "corridor")


def test_shortest_path_distances_and_pixels():
    unit = _make_sample_unit(meters_per_pixel=0.05)
    graph = build_route_graph(unit)

    # med_room (100,200) -> d2 (200,200) -> corridor (300,200) -> d1 (300,150) -> nurse_station (300,100)
    # Total px = 100 + 100 + 50 + 50 = 300 px * 0.05 = 15.0 m
    node_path, pixel_path, dist_m = shortest_path_with_pixels(
        graph, unit, "med_room", "nurse_station"
    )
    assert node_path == ["med_room", "corridor", "nurse_station"]
    assert len(pixel_path) == 5
    assert pixel_path[0] == Point(x=100, y=200)
    assert pixel_path[1] == Point(x=200, y=200)
    assert pixel_path[2] == Point(x=300, y=200)
    assert pixel_path[3] == Point(x=300, y=150)
    assert pixel_path[4] == Point(x=300, y=100)
    assert math.isclose(dist_m, 15.0, rel_tol=1e-6)

    # Same-room path
    same_nodes, same_pixels, same_dist = shortest_path_with_pixels(
        graph, unit, "med_room", "med_room"
    )
    assert same_nodes == ["med_room"]
    assert same_pixels == [Point(x=100, y=200)]
    assert same_dist == 0.0

    with pytest.raises(ValueError):
        shortest_path_with_pixels(graph, unit, "med_room", "unknown_room")


def test_schedule_parser_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    rooms = ["corridor", "nurse_station", "med_room", "lab_drop", "supply_room"]
    runs = parse_run_description(
        "pharmacy pickup every 30 minutes, specimens hourly, supply restock twice a shift",
        rooms,
    )
    assert len(runs) == 3

    by_type = {r.run_type: r for r in runs}
    assert by_type["pharmacy_pickup"].frequency_minutes == 30.0
    assert by_type["pharmacy_pickup"].origin == "med_room"
    assert by_type["pharmacy_pickup"].destination == "nurse_station"

    assert by_type["lab_specimen_drop"].frequency_minutes == 60.0
    assert by_type["lab_specimen_drop"].origin == "nurse_station"
    assert by_type["lab_specimen_drop"].destination == "lab_drop"

    assert by_type["supply_restock"].frequency_minutes == 240.0
    assert by_type["supply_restock"].origin == "supply_room"
    assert by_type["supply_restock"].destination == "nurse_station"


def test_schedule_parser_with_mocked_hydrogen(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_response = SimpleNamespace(
        text='{"runs": [{"run_type": "pharmacy_pickup", "origin": "med_room", "destination": "nurse_station", "frequency_minutes": 45.0, "label": "Meds"}]}'
    )
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    fake_genai = MagicMock()
    fake_genai.Client.return_value = mock_client
    fake_types = MagicMock()

    monkeypatch.setitem(sys.modules, "google", SimpleNamespace(genai=fake_genai))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)

    runs = parse_run_description(
        "medication run every 45 minutes",
        ["nurse_station", "med_room", "corridor"],
    )
    assert len(runs) == 1
    assert runs[0].run_type == "pharmacy_pickup"
    assert runs[0].frequency_minutes == 45.0
    assert mock_client.models.generate_content.call_args.kwargs["model"] == "gemini-4-hydrogen"


def test_simulate_shift_nurse_vs_robots():
    unit = _make_sample_unit(meters_per_pixel=0.05)
    schedule = [
        ScheduledRun(
            run_type="pharmacy_pickup",
            origin="med_room",
            destination="nurse_station",
            frequency_minutes=30.0,
        ),
        ScheduledRun(
            run_type="lab_specimen_drop",
            origin="nurse_station",
            destination="lab_drop",
            frequency_minutes=60.0,
        ),
        ScheduledRun(
            run_type="supply_restock",
            origin="supply_room",
            destination="nurse_station",
            frequency_minutes=240.0,
        ),
    ]

    comp_1 = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1)
    comp_2 = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=2)
    comp_3 = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=3)

    # 8h = 480m: 16 pharmacy (0,30..450) + 8 lab (0,60..420) + 2 supply (0,240) = 26 trips
    assert comp_1.nurse_only.total_trips == 26
    assert comp_1.nurse_only.num_robots == 0
    assert comp_1.nurse_only.nurse_minutes_walked > 0
    assert comp_1.nurse_only.robot_minutes_traveled == 0.0

    for n, comp in [(1, comp_1), (2, comp_2), (3, comp_3)]:
        assert comp.robot_assisted.num_robots == n
        assert comp.robot_assisted.total_trips == 26
        assert comp.robot_assisted.nurse_minutes_walked == 0.0
        assert comp.robot_assisted.robot_minutes_traveled > 0
        assert math.isclose(
            comp.nurse_minutes_saved, comp.nurse_only.nurse_minutes_walked, rel_tol=1e-6
        )
        assert comp.assumptions["num_robots"] == n
        assert comp.assumptions["shift_hours"] == 8.0

        for ev in comp.robot_assisted.events:
            assert ev.agent_role == "robot"
            assert ev.distance_m > 0
            assert ev.end_time_s > ev.start_time_s
            assert ev.wait_time_s >= 0.0
            assert len(ev.node_path) >= 2
            assert len(ev.pixel_path) >= 3

    # Adding robots should reduce or keep equal average delivery wait time
    assert comp_2.robot_assisted.avg_delivery_wait_s <= comp_1.robot_assisted.avg_delivery_wait_s
    assert comp_3.robot_assisted.avg_delivery_wait_s <= comp_2.robot_assisted.avg_delivery_wait_s
    assert comp_3.robot_assisted.avg_delivery_wait_s == 0.0


def test_autonomy_aware_routing_and_handoff():
    unit = _make_sample_unit(meters_per_pixel=0.05)
    # Mark supply_room as human_only, and med_room's door (d2) as narrow / robot_passable=False
    for room in unit.rooms:
        if room.id == "supply_room":
            room.autonomy_tier = "human_only"
        elif room.id == "nurse_station":
            room.autonomy_tier = "assisted"
    for door in unit.doors:
        if door.id == "d2":
            door.width_m = 0.75
            door.robot_passable = False

    robot_graph = build_route_graph(unit, robot_only=True)
    assert "supply_room" not in robot_graph.nodes
    assert not robot_graph.has_edge("med_room", "corridor")
    assert robot_graph.has_edge("nurse_station", "corridor")
    assert robot_graph.has_edge("lab_drop", "corridor")

    # Handoff for human_only supply_room should be corridor
    assert find_robot_handoff_node(unit, "supply_room") == "corridor"
    assert find_robot_handoff_node(unit, "lab_drop") == "lab_drop"
    assert find_robot_handoff_node(unit, "nonexistent") is None

    # Simulation:
    # 1) supply_room (human_only) -> nurse_station (assisted): robot hands off at corridor -> nurse_station
    # 2) med_room (blocked by narrow door d2) -> nurse_station: unreachable by robot, falls back to nurse
    schedule = [
        ScheduledRun(
            run_type="supply_restock",
            origin="supply_room",
            destination="nurse_station",
            frequency_minutes=240.0,
        ),
        ScheduledRun(
            run_type="pharmacy_pickup",
            origin="med_room",
            destination="nurse_station",
            frequency_minutes=240.0,
        ),
    ]
    comp = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1)
    assert comp.assumptions["autonomized_trips"] == 2
    assert comp.assumptions["human_only_trips"] == 2
    assert comp.robot_assisted.total_trips == 4

    robot_events = [e for e in comp.robot_assisted.events if e.agent_role == "robot"]
    nurse_events = [e for e in comp.robot_assisted.events if e.agent_role == "nurse"]
    assert len(robot_events) == 2
    assert len(nurse_events) == 2
    assert all(e.node_path == ["corridor", "nurse_station"] for e in robot_events)
    assert all(e.node_path == ["med_room", "corridor", "nurse_station"] for e in nurse_events)
    assert comp.nurse_minutes_saved > 0
    assert comp.robot_assisted.nurse_minutes_walked > 0


def test_chaotic_and_surge_scenarios_deterministic_and_multi_robot():
    unit = _make_sample_unit(meters_per_pixel=0.05)
    schedule = [
        ScheduledRun(
            run_type="pharmacy_pickup",
            origin="med_room",
            destination="nurse_station",
            frequency_minutes=30.0,
        ),
        ScheduledRun(
            run_type="lab_specimen_drop",
            origin="nurse_station",
            destination="lab_drop",
            frequency_minutes=60.0,
        ),
    ]

    chaotic_a = simulate_shift(
        unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="chaotic", seed=101
    )
    chaotic_b = simulate_shift(
        unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="chaotic", seed=101
    )
    chaotic_diff_seed = simulate_shift(
        unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="chaotic", seed=202
    )
    assert chaotic_a == chaotic_b
    assert chaotic_a.robot_assisted.events != chaotic_diff_seed.robot_assisted.events
    assert chaotic_a.assumptions["scenario_mode"] == "chaotic"
    assert chaotic_a.assumptions["stat_trips"] > 0
    assert chaotic_a.robot_assisted.total_trips > 24

    surge_1 = simulate_shift(
        unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="surge", seed=77
    )
    surge_3 = simulate_shift(
        unit, schedule, shift_hours=8.0, num_robots=3, scenario_mode="surge", seed=77
    )
    assert surge_1.assumptions["scenario_mode"] == "surge"
    assert surge_1.assumptions["stat_trips"] > chaotic_a.assumptions["stat_trips"]
    assert surge_3.robot_assisted.total_trips == surge_1.robot_assisted.total_trips

    # Multiple robots mitigate surge wait times and dispatch across robot-1..robot-3
    assert surge_3.robot_assisted.avg_delivery_wait_s <= surge_1.robot_assisted.avg_delivery_wait_s
    assert surge_3.assumptions["max_wait_s"] <= surge_1.assumptions["max_wait_s"]
    assert surge_3.assumptions["p95_wait_s"] <= surge_1.assumptions["p95_wait_s"]
    active_robots = {e.agent_id for e in surge_3.robot_assisted.events if e.agent_role == "robot"}
    assert len(active_robots) >= 2


def test_tug_pod_batching_door_clearance_and_custody_log():
    unit = _make_sample_unit(meters_per_pixel=0.05)
    # Set d2 width below 0.90m robot+pod clearance threshold even though robot_passable=True
    for door in unit.doors:
        if door.id == "d2":
            door.width_m = 0.85
            door.robot_passable = True

    robot_graph = build_route_graph(unit, robot_only=True, robot_clearance_m=0.90)
    assert not robot_graph.has_edge("med_room", "corridor")

    # Restore d2 width and test 3-compartment pod batching + custody log + 10-min flagging
    for door in unit.doors:
        if door.id == "d2":
            door.width_m = 1.10

    schedule = [
        ScheduledRun(
            run_type="pharmacy_pickup",
            origin="med_room",
            destination="nurse_station",
            frequency_minutes=480.0,
        ),
        ScheduledRun(
            run_type="lab_specimen_drop",
            origin="nurse_station",
            destination="lab_drop",
            frequency_minutes=480.0,
        ),
        ScheduledRun(
            run_type="supply_restock",
            origin="supply_room",
            destination="nurse_station",
            frequency_minutes=480.0,
        ),
    ]
    comp = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1)
    assert comp.assumptions["robot_speed_mps"] == 0.76
    assert comp.assumptions["robot_length_m"] == 0.965
    assert comp.assumptions["handoff_seconds"] == 20.0
    assert comp.assumptions["pod_compartments"] == 3
    assert comp.assumptions["batched_trips"] == 3

    # 3 runs => 3 LOAD + 3 DELIVER custody log entries
    assert len(comp.robot_assisted.custody_log) == 6
    assert comp.robot_assisted.custody_log[0].order_number == "ORD-001"
    assert comp.robot_assisted.custody_log[0].action == "LOAD"
    assert any(c.action == "DELIVER" for c in comp.robot_assisted.custody_log)
    assert all(ev.batch_id == "POD-001" for ev in comp.robot_assisted.events)


def test_world_model_stress_sweep_autonomy_plan_and_failure_modes():
    from hospital_twin.world_model import run_stress_sweep

    unit = _make_sample_unit(meters_per_pixel=0.05)
    for room in unit.rooms:
        if room.id == "supply_room":
            room.autonomy_tier = "human_only"
    for door in unit.doors:
        if door.id == "d2":
            door.width_m = 0.85

    schedule = [
        ScheduledRun(
            run_type="pharmacy_pickup",
            origin="med_room",
            destination="nurse_station",
            frequency_minutes=60.0,
        ),
        ScheduledRun(
            run_type="lab_specimen_drop",
            origin="nurse_station",
            destination="lab_drop",
            frequency_minutes=60.0,
        ),
    ]

    # Safe degradation on robot_failure and pharmacy_delay
    fail_comp = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="robot_failure", seed=7)
    assert fail_comp.robot_assisted.total_trips == 16
    pharm_comp = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="pharmacy_delay", seed=7)
    assert pharm_comp.robot_assisted.total_trips == 16

    report = run_stress_sweep(unit, schedule, shift_hours=8.0, scenario_mode="surge", seeds=(11, 22))
    assert report.runs_evaluated == 6
    assert len(report.sweep) == 3
    assert len(report.autonomy_plan) >= 4
    categories = {rec.target_id: rec.category for rec in report.autonomy_plan}
    assert categories["supply_room"] == "keep_human"
    assert categories["med_room"] == "redesign_process"
    assert len(report.ranked_bottlenecks) >= 1
    assert "ASPR-Style After-Action Summary" in report.after_action_summary
    assert "# After-Action Exercise Report" in report.after_action_markdown


def test_ems_preposition_staffing_gap_crash_cart_and_calibration():
    from hospital_twin.schemas import CalibrationInput
    from hospital_twin.world_model import calibrate_from_drill

    unit = _make_sample_unit(meters_per_pixel=0.05)
    schedule = [
        ScheduledRun(
            run_type="pharmacy_pickup",
            origin="med_room",
            destination="nurse_station",
            frequency_minutes=60.0,
        ),
        ScheduledRun(
            run_type="supply_restock",
            origin="supply_room",
            destination="nurse_station",
            frequency_minutes=120.0,
        ),
    ]

    sg = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="staffing_gap", seed=5)
    cc = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="crash_cart_missing", seed=5)
    assert sg.robot_assisted.total_trips == 12
    assert cc.robot_assisted.total_trips == 12

    surge_no_ems = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="surge", seed=9, ems_preposition=False)
    surge_ems = simulate_shift(unit, schedule, shift_hours=8.0, num_robots=1, scenario_mode="surge", seed=9, ems_preposition=True)
    assert surge_ems.robot_assisted.avg_delivery_wait_s <= surge_no_ems.robot_assisted.avg_delivery_wait_s

    cal = calibrate_from_drill(
        CalibrationInput(
            observed_nurse_speed_mps=1.15,
            observed_robot_speed_mps=0.72,
            observed_handoff_seconds=25.0,
            drill_notes="Goldstein mock drill",
        )
    )
    assert cal.nurse_speed_mps == 1.15
    assert cal.robot_speed_mps == 0.72
    assert cal.handoff_seconds == 25.0
    assert "Goldstein mock drill" in cal.delta_summary



