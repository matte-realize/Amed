"""Tests for Floor Plan Extraction & Decisions API Verification."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

from hospital_twin.extraction import (
    DECISION_MODEL,
    FALLBACK_UNIT_PATH,
    ICU_WING_FALLBACK_PATH,
    NEON_MODEL,
    compute_centroid,
    crop_door_region,
    crop_room_region,
    extract_unit_map,
    load_fallback_unit,
    verify_unit_map,
)
from hospital_twin.schemas import Door, Point, Room, UnitMap

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE_PATH = REPO_ROOT / "data" / "demo" / "sample_unit.png"
ICU_WING_IMAGE_PATH = REPO_ROOT / "data" / "demo" / "icu_wing.png"


class TestFallbackAndCropping(unittest.TestCase):
    def test_load_fallback_unit_schema_and_label(self) -> None:
        raw = json.loads(FALLBACK_UNIT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(raw.get("source"), "pre-traced fallback")

        unit = load_fallback_unit()
        self.assertIsInstance(unit, UnitMap)
        self.assertEqual(unit.image_width, 1000)
        self.assertEqual(unit.image_height, 650)
        self.assertGreaterEqual(len(unit.rooms), 10)
        self.assertGreaterEqual(len(unit.doors), 9)

        room_types = {r.type for r in unit.rooms}
        for expected_type in (
            "patient_room",
            "nurse_station",
            "medication_room",
            "supply_room",
            "lab_drop",
            "corridor",
        ):
            self.assertIn(expected_type, room_types)

        room_ids = {r.id for r in unit.rooms}
        for door in unit.doors:
            self.assertEqual(len(door.rooms_connected), 2)
            self.assertIn(door.rooms_connected[0], room_ids)
            self.assertIn(door.rooms_connected[1], room_ids)

    def test_sample_image_exists_and_matches_fallback(self) -> None:
        self.assertTrue(SAMPLE_IMAGE_PATH.exists())
        unit = load_fallback_unit()
        with Image.open(SAMPLE_IMAGE_PATH) as img:
            self.assertEqual(img.size, (unit.image_width, unit.image_height))

    def test_icu_wing_image_and_fallback_unit(self) -> None:
        self.assertTrue(ICU_WING_IMAGE_PATH.exists())
        self.assertTrue(ICU_WING_FALLBACK_PATH.exists())

        raw = json.loads(ICU_WING_FALLBACK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(raw.get("source"), "pre-traced fallback")

        icu = load_fallback_unit("icu_wing")
        default_unit = load_fallback_unit()
        self.assertIsInstance(icu, UnitMap)
        self.assertEqual((icu.image_width, icu.image_height), (1000, 650))
        with Image.open(ICU_WING_IMAGE_PATH) as img:
            self.assertEqual(img.size, (icu.image_width, icu.image_height))

        # Distinct from default sample_unit fallback
        icu_ids = {r.id for r in icu.rooms}
        default_ids = {r.id for r in default_unit.rooms}
        self.assertNotEqual(icu_ids, default_ids)

        # Verify required L-wing rooms & autonomy tiers
        rooms_by_id = {r.id: r for r in icu.rooms}
        self.assertIn("corridor_main", rooms_by_id)
        self.assertIn("corridor_east", rooms_by_id)
        self.assertEqual(rooms_by_id["central_nurse_station"].autonomy_tier, "assisted")
        self.assertEqual(rooms_by_id["pharmacy_satellite"].autonomy_tier, "autonomous")
        self.assertEqual(rooms_by_id["clean_supply"].autonomy_tier, "autonomous")
        self.assertEqual(rooms_by_id["lab_pneumatic_drop"].autonomy_tier, "autonomous")

        icu_rooms = [r for r in icu.rooms if r.type == "patient_room"]
        self.assertEqual(len(icu_rooms), 4)
        self.assertTrue(all(r.autonomy_tier == "human_only" for r in icu_rooms))

        # Verify 2 narrow 0.82m non-passable airlock doors and all doors reference valid rooms
        narrow_doors = [d for d in icu.doors if not d.robot_passable]
        self.assertEqual(len(narrow_doors), 2)
        for d in narrow_doors:
            self.assertAlmostEqual(d.width_m, 0.82, places=2)
        for d in icu.doors:
            self.assertEqual(len(d.rooms_connected), 2)
            self.assertIn(d.rooms_connected[0], rooms_by_id)
            self.assertIn(d.rooms_connected[1], rooms_by_id)

        # Unknown plan_id gracefully falls back to default sample_unit
        unknown_fallback = load_fallback_unit("nonexistent_plan")
        self.assertEqual({r.id for r in unknown_fallback.rooms}, default_ids)

    def test_crop_room_and_door_regions(self) -> None:
        unit = load_fallback_unit()
        with Image.open(SAMPLE_IMAGE_PATH) as img:
            room_crop = crop_room_region(img, unit.rooms[0], padding=16)
            self.assertGreater(room_crop.width, 0)
            self.assertGreater(room_crop.height, 0)
            self.assertLessEqual(room_crop.width, img.width)
            self.assertLessEqual(room_crop.height, img.height)

            door_crop = crop_door_region(img, unit.doors[0], radius=40)
            self.assertEqual(door_crop.size, (80, 80))

            # Edge-clamped door near (0, 0)
            edge_door = Door(id="d_edge", position=Point(x=5, y=5), rooms_connected=["room_101", "corridor_main"])
            edge_crop = crop_door_region(img, edge_door, radius=40)
            self.assertEqual(edge_crop.size, (45, 45))

    def test_compute_centroid(self) -> None:
        poly = [Point(x=0, y=0), Point(x=100, y=0), Point(x=100, y=200), Point(x=0, y=200)]
        c = compute_centroid(poly)
        self.assertAlmostEqual(c.x, 50.0)
        self.assertAlmostEqual(c.y, 100.0)


class TestMockedGenAIExtractionAndVerification(unittest.TestCase):
    def test_extract_unit_map_with_mocked_neon(self) -> None:
        mock_payload = {
            "rooms": [
                {
                    "id": "room_101",
                    "label": "Room 101",
                    "type": "patient_room",
                    "polygon": [
                        {"x": 40, "y": 40},
                        {"x": 220, "y": 40},
                        {"x": 220, "y": 240},
                        {"x": 40, "y": 240},
                    ],
                },
                {
                    "id": "corridor_main",
                    "label": "Main Corridor",
                    "type": "corridor",
                    "polygon": [
                        {"x": 40, "y": 240},
                        {"x": 960, "y": 240},
                        {"x": 960, "y": 380},
                        {"x": 40, "y": 380},
                    ],
                },
            ],
            "doors": [
                {
                    "id": "door_101",
                    "position": {"x": 130, "y": 240},
                    "rooms_connected": ["room_101", "corridor_main"],
                }
            ],
        }

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = SimpleNamespace(
            parsed=mock_payload,
            text=json.dumps(mock_payload),
        )

        with patch("hospital_twin.extraction._get_client", return_value=mock_client):
            unit = extract_unit_map(str(SAMPLE_IMAGE_PATH), meters_per_pixel=0.05)

        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], NEON_MODEL)

        self.assertEqual(unit.image_width, 1000)
        self.assertEqual(unit.image_height, 650)
        self.assertEqual(len(unit.rooms), 2)
        self.assertEqual(len(unit.doors), 1)
        # Centroid auto-computed when omitted
        self.assertIsNotNone(unit.rooms[0].centroid)
        self.assertAlmostEqual(unit.rooms[0].centroid.x, 130.0)
        self.assertAlmostEqual(unit.rooms[0].centroid.y, 140.0)

    def test_verify_unit_map_keep_reextract_and_drop(self) -> None:
        unit = UnitMap(
            image_width=1000,
            image_height=650,
            meters_per_pixel=0.05,
            rooms=[
                Room(
                    id="room_101",
                    label="Room 101",
                    type="patient_room",
                    polygon=[Point(x=40, y=40), Point(x=220, y=40), Point(x=220, y=240), Point(x=40, y=240)],
                ),
                Room(
                    id="nurse_station",
                    label="Station?",
                    type="other",
                    polygon=[Point(x=400, y=40), Point(x=620, y=40), Point(x=620, y=240), Point(x=400, y=240)],
                ),
                Room(
                    id="hallucinated_room",
                    label="Helipad",
                    type="other",
                    polygon=[Point(x=10, y=10), Point(x=30, y=10), Point(x=30, y=30), Point(x=10, y=30)],
                ),
            ],
            doors=[
                Door(id="door_101", position=Point(x=130, y=240), rooms_connected=["room_101", "nurse_station"]),
                Door(id="door_bad", position=Point(x=20, y=20), rooms_connected=["room_101", "hallucinated_room"]),
            ],
        )

        def fake_decision_create(*, model, input, questions):
            text = input[1]["text"]
            if "is_room_label" in questions:
                if "Room 101" in text:
                    return SimpleNamespace(answers={"is_room_label": SimpleNamespace(noul=0.92, confidence=0.92)})
                if "Station?" in text:
                    return SimpleNamespace(answers={"is_room_label": SimpleNamespace(noul=0.45, confidence=0.55)})
                if "Nurse Station" in text:
                    return SimpleNamespace(answers={"is_room_label": SimpleNamespace(noul=0.88, confidence=0.88)})
                if "Helipad" in text:
                    return SimpleNamespace(answers={"is_room_label": SimpleNamespace(noul=0.12, confidence=0.88)})
            if "is_door" in questions:
                if "door_101" in text:
                    return SimpleNamespace(answers={"is_door": SimpleNamespace(noul=0.85, confidence=0.85)})
                if "door_bad" in text:
                    return SimpleNamespace(answers={"is_door": SimpleNamespace(noul=0.80, confidence=0.80)})
            raise AssertionError(f"Unexpected decision request: {text}")

        mock_decision_client = MagicMock()
        mock_decision_client.decisions.create.side_effect = fake_decision_create

        mock_neon_client = MagicMock()
        mock_neon_client.models.generate_content.return_value = SimpleNamespace(
            parsed={"label": "Nurse Station", "type": "nurse_station"},
            text='{"label": "Nurse Station", "type": "nurse_station"}',
        )

        def fake_get_client(api_version=None):
            return mock_decision_client if api_version == "v1beta" else mock_neon_client

        with patch("hospital_twin.extraction._get_client", side_effect=fake_get_client):
            verified, stats = verify_unit_map(str(SAMPLE_IMAGE_PATH), unit)

        self.assertEqual(len(verified.rooms), 2)
        self.assertEqual(verified.rooms[1].label, "Nurse Station")
        self.assertEqual(verified.rooms[1].type, "nurse_station")
        self.assertEqual(len(verified.doors), 1)
        self.assertEqual(verified.doors[0].id, "door_101")

        self.assertEqual(stats["kept"], 3)
        self.assertEqual(stats["dropped"], 2)
        self.assertEqual(stats["re_extracted"], 1)
        self.assertAlmostEqual(stats["mean_confidence"], round((0.92 + 0.88 + 0.85) / 3, 4), places=4)

        first_call_kwargs = mock_decision_client.decisions.create.call_args_list[0].kwargs
        self.assertEqual(first_call_kwargs["model"], DECISION_MODEL)


if __name__ == "__main__":
    unittest.main()
