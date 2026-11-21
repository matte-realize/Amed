"""SimPy discrete-event shift simulation comparing nurse-only vs. robot-assisted runs."""

from __future__ import annotations

from typing import Literal

import simpy

from hospital_twin.routing import build_route_graph, shortest_path_with_pixels
from hospital_twin.schemas import (
    Point,
    ScheduledRun,
    ShiftSummary,
    SimulationComparison,
    TripEvent,
    UnitMap,
)

DEFAULT_NUM_NURSES = 2


def _run_single_mode(
    routes: list[tuple[ScheduledRun, list[str], list[Point], float]],
    mode: Literal["nurse_only", "robot_assisted"],
    shift_duration_s: float,
    num_robots: int,
    nurse_speed_mps: float,
    robot_speed_mps: float,
) -> ShiftSummary:
    env = simpy.Environment()
    events: list[TripEvent] = []

    use_robots = mode == "robot_assisted" and num_robots > 0
    if use_robots:
        agent_role: Literal["nurse", "robot"] = "robot"
        speed_mps = robot_speed_mps
        agent_ids = [f"robot-{i + 1}" for i in range(num_robots)]
    else:
        agent_role = "nurse"
        speed_mps = nurse_speed_mps
        agent_ids = [f"nurse-{i + 1}" for i in range(DEFAULT_NUM_NURSES)]

    pool = simpy.Store(env, capacity=len(agent_ids))
    for aid in agent_ids:
        pool.put(aid)

    def execute_trip(
        run: ScheduledRun,
        node_path: list[str],
        pixel_path: list[Point],
        distance_m: float,
    ):
        requested_at = env.now
        agent_id = yield pool.get()
        start_time_s = env.now
        wait_time_s = start_time_s - requested_at
        travel_time_s = distance_m / speed_mps if distance_m > 0 else 0.0
        yield env.timeout(travel_time_s)
        end_time_s = env.now
        yield pool.put(agent_id)

        events.append(
            TripEvent(
                agent_id=agent_id,
                agent_role=agent_role,
                run_type=run.run_type,
                start_time_s=round(start_time_s, 3),
                end_time_s=round(end_time_s, 3),
                wait_time_s=round(wait_time_s, 3),
                distance_m=round(distance_m, 3),
                node_path=node_path,
                pixel_path=pixel_path,
            )
        )

    def schedule_generator(
        run: ScheduledRun,
        node_path: list[str],
        pixel_path: list[Point],
        distance_m: float,
    ):
        period_s = max(60.0, run.frequency_minutes * 60.0)
        t = 0.0
        while t < shift_duration_s:
            env.process(execute_trip(run, node_path, pixel_path, distance_m))
            next_t = t + period_s
            if next_t >= shift_duration_s:
                break
            yield env.timeout(period_s)
            t = next_t

    for run, node_path, pixel_path, distance_m in routes:
        env.process(schedule_generator(run, node_path, pixel_path, distance_m))

    # Run until all trips dispatched within the shift have completed.
    env.run()

    events.sort(key=lambda e: (e.start_time_s, e.end_time_s, e.agent_id))

    nurse_minutes_walked = sum(
        (e.end_time_s - e.start_time_s) / 60.0 for e in events if e.agent_role == "nurse"
    )
    robot_minutes_traveled = sum(
        (e.end_time_s - e.start_time_s) / 60.0 for e in events if e.agent_role == "robot"
    )
    avg_wait_s = sum(e.wait_time_s for e in events) / len(events) if events else 0.0

    return ShiftSummary(
        mode=mode,
        num_robots=num_robots if mode == "robot_assisted" else 0,
        shift_duration_s=round(shift_duration_s, 3),
        nurse_minutes_walked=round(nurse_minutes_walked, 3),
        robot_minutes_traveled=round(robot_minutes_traveled, 3),
        avg_delivery_wait_s=round(avg_wait_s, 3),
        total_trips=len(events),
        events=events,
    )


def simulate_shift(
    unit: UnitMap,
    schedule: list[ScheduledRun],
    shift_hours: float = 8.0,
    num_robots: int = 1,
    nurse_speed_mps: float = 1.2,
    robot_speed_mps: float = 0.8,
) -> SimulationComparison:
    """Simulate a hospital shift in nurse-only and robot-assisted modes using SimPy."""
    if shift_hours <= 0:
        raise ValueError("shift_hours must be positive.")
    if num_robots < 0:
        raise ValueError("num_robots must be non-negative.")
    if nurse_speed_mps <= 0 or robot_speed_mps <= 0:
        raise ValueError("nurse_speed_mps and robot_speed_mps must be positive.")

    graph = build_route_graph(unit)
    routes: list[tuple[ScheduledRun, list[str], list[Point], float]] = []
    for run in schedule:
        node_path, pixel_path, distance_m = shortest_path_with_pixels(
            graph, unit, run.origin, run.destination
        )
        routes.append((run, node_path, pixel_path, distance_m))

    shift_duration_s = shift_hours * 3600.0

    nurse_only = _run_single_mode(
        routes=routes,
        mode="nurse_only",
        shift_duration_s=shift_duration_s,
        num_robots=0,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
    )
    robot_assisted = _run_single_mode(
        routes=routes,
        mode="robot_assisted",
        shift_duration_s=shift_duration_s,
        num_robots=num_robots,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
    )

    nurse_minutes_saved = round(
        nurse_only.nurse_minutes_walked - robot_assisted.nurse_minutes_walked, 3
    )

    assumptions: dict[str, float | str] = {
        "shift_hours": shift_hours,
        "num_nurses": DEFAULT_NUM_NURSES,
        "num_robots": num_robots,
        "nurse_speed_mps": nurse_speed_mps,
        "robot_speed_mps": robot_speed_mps,
        "meters_per_pixel": unit.meters_per_pixel,
        "trip_model": (
            "One-way shortest path via room centroids and doors; "
            "courier runs are offloaded to robots in robot_assisted mode."
        ),
    }

    return SimulationComparison(
        assumptions=assumptions,
        nurse_only=nurse_only,
        robot_assisted=robot_assisted,
        nurse_minutes_saved=nurse_minutes_saved,
    )
