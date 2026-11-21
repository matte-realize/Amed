"""SimPy discrete-event shift simulation comparing nurse-only vs. robot-assisted runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import simpy

from hospital_twin.routing import (
    build_route_graph,
    find_robot_handoff_node,
    shortest_path_with_pixels,
)
from hospital_twin.schemas import (
    Point,
    ScheduledRun,
    ShiftSummary,
    SimulationComparison,
    TripEvent,
    UnitMap,
)

DEFAULT_NUM_NURSES = 2


@dataclass(frozen=True)
class _PlannedRoute:
    run: ScheduledRun
    agent_role: Literal["nurse", "robot"]
    node_path: list[str]
    pixel_path: list[Point]
    distance_m: float


def _plan_robot_assisted_route(
    unit: UnitMap,
    full_graph: nx.Graph,
    robot_graph: nx.Graph,
    run: ScheduledRun,
    num_robots: int,
) -> _PlannedRoute:
    if num_robots > 0:
        handoff_origin = find_robot_handoff_node(unit, run.origin)
        handoff_dest = find_robot_handoff_node(unit, run.destination)
        if (
            handoff_origin is not None
            and handoff_dest is not None
            and handoff_origin in robot_graph
            and handoff_dest in robot_graph
            and (handoff_origin != handoff_dest or run.origin == run.destination)
        ):
            try:
                node_path, pixel_path, distance_m = shortest_path_with_pixels(
                    robot_graph, unit, handoff_origin, handoff_dest
                )
                return _PlannedRoute(
                    run=run,
                    agent_role="robot",
                    node_path=node_path,
                    pixel_path=pixel_path,
                    distance_m=distance_m,
                )
            except ValueError:
                pass

    node_path, pixel_path, distance_m = shortest_path_with_pixels(
        full_graph, unit, run.origin, run.destination
    )
    return _PlannedRoute(
        run=run,
        agent_role="nurse",
        node_path=node_path,
        pixel_path=pixel_path,
        distance_m=distance_m,
    )


def _run_single_mode(
    planned_routes: list[_PlannedRoute],
    mode: Literal["nurse_only", "robot_assisted"],
    shift_duration_s: float,
    num_robots: int,
    nurse_speed_mps: float,
    robot_speed_mps: float,
) -> ShiftSummary:
    env = simpy.Environment()
    events: list[TripEvent] = []

    nurse_pool = simpy.Store(env, capacity=DEFAULT_NUM_NURSES)
    for i in range(DEFAULT_NUM_NURSES):
        nurse_pool.put(f"nurse-{i + 1}")

    robot_pool: simpy.Store | None = None
    if mode == "robot_assisted" and num_robots > 0:
        robot_pool = simpy.Store(env, capacity=num_robots)
        for i in range(num_robots):
            robot_pool.put(f"robot-{i + 1}")

    def execute_trip(route: _PlannedRoute):
        use_robot = route.agent_role == "robot" and robot_pool is not None
        pool = robot_pool if use_robot else nurse_pool
        role: Literal["nurse", "robot"] = "robot" if use_robot else "nurse"
        speed_mps = robot_speed_mps if use_robot else nurse_speed_mps

        requested_at = env.now
        agent_id = yield pool.get()
        start_time_s = env.now
        wait_time_s = start_time_s - requested_at
        travel_time_s = route.distance_m / speed_mps if route.distance_m > 0 else 0.0
        yield env.timeout(travel_time_s)
        end_time_s = env.now
        yield pool.put(agent_id)

        events.append(
            TripEvent(
                agent_id=agent_id,
                agent_role=role,
                run_type=route.run.run_type,
                start_time_s=round(start_time_s, 3),
                end_time_s=round(end_time_s, 3),
                wait_time_s=round(wait_time_s, 3),
                distance_m=round(route.distance_m, 3),
                node_path=route.node_path,
                pixel_path=route.pixel_path,
            )
        )

    def schedule_generator(route: _PlannedRoute):
        period_s = max(60.0, route.run.frequency_minutes * 60.0)
        t = 0.0
        while t < shift_duration_s:
            env.process(execute_trip(route))
            next_t = t + period_s
            if next_t >= shift_duration_s:
                break
            yield env.timeout(period_s)
            t = next_t

    for route in planned_routes:
        env.process(schedule_generator(route))

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

    full_graph = build_route_graph(unit, robot_only=False)
    robot_graph = build_route_graph(unit, robot_only=True)

    nurse_routes: list[_PlannedRoute] = []
    robot_assisted_routes: list[_PlannedRoute] = []
    for run in schedule:
        node_path, pixel_path, distance_m = shortest_path_with_pixels(
            full_graph, unit, run.origin, run.destination
        )
        nurse_routes.append(
            _PlannedRoute(
                run=run,
                agent_role="nurse",
                node_path=node_path,
                pixel_path=pixel_path,
                distance_m=distance_m,
            )
        )
        robot_assisted_routes.append(
            _plan_robot_assisted_route(unit, full_graph, robot_graph, run, num_robots)
        )

    shift_duration_s = shift_hours * 3600.0

    nurse_only = _run_single_mode(
        planned_routes=nurse_routes,
        mode="nurse_only",
        shift_duration_s=shift_duration_s,
        num_robots=0,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
    )
    robot_assisted = _run_single_mode(
        planned_routes=robot_assisted_routes,
        mode="robot_assisted",
        shift_duration_s=shift_duration_s,
        num_robots=num_robots,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
    )

    nurse_minutes_saved = round(
        nurse_only.nurse_minutes_walked - robot_assisted.nurse_minutes_walked, 3
    )
    autonomized_trips = sum(1 for e in robot_assisted.events if e.agent_role == "robot")
    human_only_trips = sum(1 for e in robot_assisted.events if e.agent_role == "nurse")

    assumptions: dict[str, float | str] = {
        "shift_hours": shift_hours,
        "num_nurses": DEFAULT_NUM_NURSES,
        "num_robots": num_robots,
        "nurse_speed_mps": nurse_speed_mps,
        "robot_speed_mps": robot_speed_mps,
        "meters_per_pixel": unit.meters_per_pixel,
        "autonomized_trips": autonomized_trips,
        "human_only_trips": human_only_trips,
        "trip_model": (
            "Autonomy-aware shortest path via room centroids and doors; "
            "human_only rooms use nearest autonomous/assisted handoff nodes, "
            "and robot-unreachable runs fall back to nurses."
        ),
    }

    return SimulationComparison(
        assumptions=assumptions,
        nurse_only=nurse_only,
        robot_assisted=robot_assisted,
        nurse_minutes_saved=nurse_minutes_saved,
    )
