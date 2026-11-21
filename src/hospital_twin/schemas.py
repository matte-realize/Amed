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
    autonomy_tier: Literal["autonomous", "assisted", "human_only"] = "assisted"
    autonomy_reason: str = ""
    proposed_solution: str = ""
    prerequisites: list[str] = Field(default_factory=list)
    obstacles: list[str] = Field(default_factory=list)


class Door(BaseModel):
    id: str
    position: Point
    rooms_connected: list[str] = Field(min_length=2, max_length=2)
    confidence: Optional[float] = None
    width_m: float = 1.0
    robot_passable: bool = True


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
    batch_id: Optional[str] = None


class CustodyLogEntry(BaseModel):
    order_number: str
    who: str
    where: str
    when_s: float
    when_formatted: str
    action: Literal["LOAD", "DELIVER", "FLAG_UNCOLLECTED"]
    run_type: RunType
    flagged_uncollected: bool = False


class ShiftSummary(BaseModel):
    mode: Literal["nurse_only", "robot_assisted"]
    num_robots: int = 0
    shift_duration_s: float
    nurse_minutes_walked: float
    robot_minutes_traveled: float
    avg_delivery_wait_s: float
    total_trips: int
    events: list[TripEvent]
    custody_log: list[CustodyLogEntry] = Field(default_factory=list)
    flagged_exceptions: int = 0


class SimulationComparison(BaseModel):
    assumptions: dict[str, float | str]
    nurse_only: ShiftSummary
    robot_assisted: ShiftSummary
    nurse_minutes_saved: float


class AutonomyRecommendation(BaseModel):
    target_id: str
    target_label: str
    category: Literal["use_robot", "redesign_process", "keep_human", "research_gap"]
    rationale: str
    action: str = ""


class BottleneckItem(BaseModel):
    rank: int
    source: str
    impact_minutes: float
    detail: str


class StressTestSweepPoint(BaseModel):
    num_robots: int
    mean_nurse_minutes_saved: float
    mean_p95_wait_s: float
    max_flagged_exceptions: int


class StressTestReport(BaseModel):
    scenario_mode: str
    runs_evaluated: int
    tipping_point: str
    sweep: list[StressTestSweepPoint]
    autonomy_plan: list[AutonomyRecommendation]
    ranked_bottlenecks: list[BottleneckItem]
    after_action_summary: str
    after_action_markdown: str = ""


class CalibrationInput(BaseModel):
    observed_nurse_speed_mps: Optional[float] = Field(default=None, gt=0.1, le=5.0)
    observed_robot_speed_mps: Optional[float] = Field(default=None, gt=0.1, le=5.0)
    observed_handoff_seconds: Optional[float] = Field(default=None, ge=0.0, le=300.0)
    drill_notes: str = ""


class CalibrationResult(BaseModel):
    nurse_speed_mps: float
    robot_speed_mps: float
    handoff_seconds: float
    delta_summary: str
