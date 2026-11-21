"""Shared data contracts for the 5-stage Hospital Twin pipeline."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


RoomType = Literal[
    "patient_room",
    "nurse_station",
    "medication_room",
    "supply_room",
    "lab_drop",
    "corridor",
    "other",
]

RunType = Literal["pharmacy_pickup", "lab_specimen_drop", "supply_restock"]


class Point(BaseModel):
    x: float
    y: float


class Room(BaseModel):
    id: str
    label: str
    type: RoomType
    polygon: list[Point]
    centroid: Optional[Point] = None
    confidence: Optional[float] = None


class Door(BaseModel):
    id: str
    position: Point
    rooms_connected: list[str] = Field(min_length=2, max_length=2)
    confidence: Optional[float] = None


class UnitMap(BaseModel):
    image_width: int
    image_height: int
    meters_per_pixel: float = 0.05
    rooms: list[Room]
    doors: list[Door]


class ScheduledRun(BaseModel):
    run_type: RunType
    origin: str
    destination: str
    frequency_minutes: float
    label: Optional[str] = None


class TripEvent(BaseModel):
    agent_id: str
    agent_role: Literal["nurse", "robot"]
    run_type: RunType
    start_time_s: float
    end_time_s: float
    wait_time_s: float = 0.0
    distance_m: float
    node_path: list[str]
    pixel_path: list[Point]


class ShiftSummary(BaseModel):
    mode: Literal["nurse_only", "robot_assisted"]
    num_robots: int = 0
    shift_duration_s: float
    nurse_minutes_walked: float
    robot_minutes_traveled: float
    avg_delivery_wait_s: float
    total_trips: int
    events: list[TripEvent]


class SimulationComparison(BaseModel):
    assumptions: dict[str, float | str]
    nurse_only: ShiftSummary
    robot_assisted: ShiftSummary
    nurse_minutes_saved: float
