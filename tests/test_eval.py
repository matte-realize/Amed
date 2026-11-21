"""Unit tests for Hospital Twin evaluation scoring math and dataset/harness logic."""

from __future__ import annotations

import json
from pathlib import Path

from hospital_twin.eval_proof import (
    ensure_synthetic_eval_dataset,
    format_markdown_summary,
    labels_match,
    load_eval_dataset,
    parse_cubicasa_svg,
    point_in_polygon,
    polygon_centroid,
    polygon_iou,
    run_evaluation,
    score_unit_map,
)
from hospital_twin.schemas import Door, Point, Room, UnitMap


def _rect(x1: float, y1: float, x2: float, y2: float) -> list[Point]:
    return [
        Point(x=x1, y=y1),
        Point(x=x2, y=y1),
        Point(x=x2, y=y2),
        Point(x=x1, y=y2),
    ]


def test_polygon_iou_identical_and_disjoint() -> None:
    a = _rect(0, 0, 100, 100)
    b = _rect(0, 0, 100, 100)
    c = _rect(200, 200, 300, 300)

    assert polygon_iou(a, b) > 0.99
    assert polygon_iou(a, c) == 0.0


def test_polygon_iou_partial_overlap() -> None:
    a = _rect(0, 0, 100, 100)
    b = _rect(50, 0, 150, 100)
    # Exact IoU is 5000 / 15000 = 0.3333
    iou = polygon_iou(a, b, grid_size=60)
    assert 0.28 <= iou <= 0.38


def test_point_in_polygon_concave_l_shape() -> None:
    l_shape = [
        Point(x=0, y=0),
        Point(x=100, y=0),
        Point(x=100, y=40),
        Point(x=40, y=40),
        Point(x=40, y=100),
        Point(x=0, y=100),
    ]
    assert point_in_polygon(Point(x=20, y=80), l_shape) is True
    assert point_in_polygon(Point(x=80, y=80), l_shape) is False
    c = polygon_centroid(l_shape)
    assert 0 < c.x < 100 and 0 < c.y < 100


def test_centroid_in_polygon_fallback_when_iou_is_low() -> None:
    gt = UnitMap(
        image_width=400,
        image_height=400,
        rooms=[
            Room(
                id="r1",
                label="Nurse Station",
                type="nurse_station",
                polygon=_rect(0, 0, 200, 200),
            )
        ],
        doors=[],
    )
    # Small box (20x20 -> IoU = 400/40000 = 0.01 < 0.3) centered inside gt room.
    pred = UnitMap(
        image_width=400,
        image_height=400,
        rooms=[
            Room(
                id="p1",
                label="nurse_station",
                type="nurse_station",
                polygon=_rect(90, 90, 110, 110),
            )
        ],
        doors=[],
    )
    metrics = score_unit_map(gt, pred, iou_threshold=0.3)
    assert metrics.matched_rooms == 1
    assert metrics.room_detection_rate == 1.0
    assert metrics.room_label_accuracy == 1.0


def test_label_accuracy_and_normalization() -> None:
    assert labels_match("Patient Room 101", "patient_room-101") is True
    assert labels_match("Medication Room", "Supply Room") is False

    gt = UnitMap(
        image_width=400,
        image_height=200,
        rooms=[
            Room(id="r1", label="Nurse Station", type="nurse_station", polygon=_rect(0, 0, 100, 100)),
            Room(id="r2", label="Lab Drop", type="lab_drop", polygon=_rect(120, 0, 220, 100)),
        ],
        doors=[],
    )
    pred = UnitMap(
        image_width=400,
        image_height=200,
        rooms=[
            Room(id="p1", label="nurse station", type="nurse_station", polygon=_rect(0, 0, 100, 100)),
            Room(id="p2", label="Supply Room", type="supply_room", polygon=_rect(120, 0, 220, 100)),
        ],
        doors=[],
    )
    metrics = score_unit_map(gt, pred)
    assert metrics.room_detection_rate == 1.0
    assert metrics.room_label_accuracy == 0.5


def test_door_proximity_matching_one_to_one() -> None:
    gt = UnitMap(
        image_width=400,
        image_height=200,
        rooms=[
            Room(id="r1", label="A", type="corridor", polygon=_rect(0, 0, 100, 100)),
            Room(id="r2", label="B", type="patient_room", polygon=_rect(100, 0, 200, 100)),
        ],
        doors=[
            Door(id="d1", position=Point(x=100, y=50), rooms_connected=["r1", "r2"]),
            Door(id="d2", position=Point(x=200, y=50), rooms_connected=["r1", "r2"]),
        ],
    )
    # Two predictions near d1, none near d2 -> only 1 GT door matched.
    pred = UnitMap(
        image_width=400,
        image_height=200,
        rooms=gt.rooms,
        doors=[
            Door(id="pd1", position=Point(x=103, y=54), rooms_connected=["r1", "r2"]),
            Door(id="pd2", position=Point(x=105, y=55), rooms_connected=["r1", "r2"]),
        ],
    )
    metrics = score_unit_map(gt, pred, door_radius_px=25.0)
    assert metrics.matched_doors == 1
    assert metrics.door_detection_rate == 0.5
    assert metrics.door_precision == 0.5


def test_cubicasa_svg_loader(tmp_path: Path) -> None:
    svg_content = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">
      <g class="Space Bedroom" id="room_bed">
        <polygon points="10,10 190,10 190,190 10,190" />
      </g>
      <g class="Space Hall" id="room_hall">
        <polygon points="190,10 380,10 380,190 190,190" />
      </g>
      <g class="Door Swing" id="door_1">
        <polygon points="185,90 195,90 195,110 185,110" />
      </g>
    </svg>"""
    plan_dir = tmp_path / "cubi_001"
    plan_dir.mkdir()
    (plan_dir / "model.svg").write_text(svg_content, encoding="utf-8")
    (plan_dir / "F1_scaled.png").write_bytes(b"fake-png")

    unit = parse_cubicasa_svg(plan_dir / "model.svg", image_width=400, image_height=300)
    assert len(unit.rooms) == 2
    assert unit.rooms[0].type == "patient_room"
    assert unit.rooms[1].type == "corridor"
    assert len(unit.doors) == 1
    assert set(unit.doors[0].rooms_connected) == {"room_bed", "room_hall"}

    samples = load_eval_dataset(tmp_path)
    assert len(samples) == 1
    assert samples[0].source_format == "cubicasa5k_svg"


def test_run_evaluation_outputs_json_and_markdown(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    samples = ensure_synthetic_eval_dataset(eval_dir)
    assert len(samples) >= 2

    out_json = eval_dir / "eval_results.json"
    results = run_evaluation(eval_dir=eval_dir, output_path=out_json)

    assert out_json.exists()
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["num_plans"] == len(samples)
    assert loaded["aggregate"]["verified"]["room_detection_rate"] >= loaded["aggregate"]["baseline"]["room_detection_rate"]
    md = format_markdown_summary(loaded)
    assert "Baseline (Neon only)" in md
    assert "Verified (Neon + Decisions API)" in md
