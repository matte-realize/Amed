"""SimPy discrete-event shift simulation comparing nurse-only vs. robot-assisted runs."""

from __future__ import annotations

import math
import random
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
    RunType,
    ScheduledRun,
    ShiftSummary,
    SimulationComparison,
    TripEvent,
    UnitMap,
)

DEFAULT_NUM_NURSES = 2
ScenarioMode = Literal["routine", "chaotic", "surge"]


@dataclass(frozen=True)
class _PlannedRoute:
    run: ScheduledRun
    agent_role: Literal["nurse", "robot"]
    node_path: list[str]
    pixel_path: list[Point]
    distance_m: float


@dataclass(frozen=True)
class _TripRequest:
    arrival_time_s: float
    nurse_route: _PlannedRoute
    robot_route: _PlannedRoute
    speed_factor: float = 1.0
    delay_s: float = 0.0
    is_stat: bool = False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


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


def _generate_trip_requests(
    unit: UnitMap,
    full_graph: nx.Graph,
    robot_graph: nx.Graph,
    schedule: list[ScheduledRun],
    shift_duration_s: float,
    num_robots: int,
    scenario_mode: ScenarioMode,
    seed: int,
) -> list[_TripRequest]:
    rng = random.Random(seed)
    requests: list[_TripRequest] = []

    for run in schedule:
        nurse_node_path, nurse_pixel_path, nurse_dist_m = shortest_path_with_pixels(
            full_graph, unit, run.origin, run.destination
        )
        nurse_route = _PlannedRoute(
            run=run,
            agent_role="nurse",
            node_path=nurse_node_path,
            pixel_path=nurse_pixel_path,
            distance_m=nurse_dist_m,
        )
        robot_route = _plan_robot_assisted_route(
            unit, full_graph, robot_graph, run, num_robots
        )

        period_s = max(60.0, run.frequency_minutes * 60.0)
        t = 0.0
        while t < shift_duration_s:
            if scenario_mode == "routine":
                arrival_s = t
                speed_factor = 1.0
                delay_s = 0.0
            else:
                jitter = rng.uniform(-0.20 * period_s, 0.25 * period_s)
                arrival_s = min(max(0.0, t + jitter), max(0.0, shift_duration_s - 1.0))
                speed_factor = rng.uniform(0.82, 1.18)
                delay_prob = 0.20 if scenario_mode == "chaotic" else 0.30
                max_delay = 35.0 if scenario_mode == "chaotic" else 50.0
                delay_s = rng.uniform(8.0, max_delay) if rng.random() < delay_prob else 0.0

            requests.append(
                _TripRequest(
                    arrival_time_s=arrival_s,
                    nurse_route=nurse_route,
                    robot_route=robot_route,
                    speed_factor=speed_factor,
                    delay_s=delay_s,
                    is_stat=False,
                )
            )
            t += period_s

    if scenario_mode in ("chaotic", "surge"):
        requests.extend(
            _generate_stat_bursts(
                unit=unit,
                full_graph=full_graph,
                robot_graph=robot_graph,
                shift_duration_s=shift_duration_s,
                num_robots=num_robots,
                scenario_mode=scenario_mode,
                rng=rng,
            )
        )

    requests.sort(key=lambda r: r.arrival_time_s)
    return requests


def _generate_stat_bursts(
    unit: UnitMap,
    full_graph: nx.Graph,
    robot_graph: nx.Graph,
    shift_duration_s: float,
    num_robots: int,
    scenario_mode: ScenarioMode,
    rng: random.Random,
) -> list[_TripRequest]:
    non_corridor = [r.id for r in unit.rooms if r.type != "corridor" and r.id in full_graph]
    candidate_rooms = non_corridor if len(non_corridor) >= 2 else list(full_graph.nodes)
    if len(candidate_rooms) < 2:
        return []

    reachable_pairs: list[tuple[str, str]] = []
    for src in candidate_rooms:
        for dst in candidate_rooms:
            if src != dst and nx.has_path(full_graph, src, dst):
                reachable_pairs.append((src, dst))
    if not reachable_pairs:
        return []

    shift_hours = shift_duration_s / 3600.0
    if scenario_mode == "chaotic":
        total_stat = max(4, int(round(shift_hours * 1.5)))
        num_clusters = max(2, total_stat // 3)
        delay_prob = 0.25
        max_delay = 40.0
    else:  # surge
        total_stat = max(8, int(round(shift_hours * 3.0)))
        num_clusters = max(3, total_stat // 4)
        delay_prob = 0.35
        max_delay = 55.0

    cluster_centers = sorted(
        rng.uniform(0.05 * shift_duration_s, 0.90 * shift_duration_s)
        for _ in range(num_clusters)
    )
    run_types: tuple[RunType, ...] = (
        "lab_specimen_drop",
        "pharmacy_pickup",
        "supply_restock",
    )

    stat_requests: list[_TripRequest] = []
    for idx in range(total_stat):
        center = cluster_centers[idx % num_clusters]
        arrival_s = min(
            max(0.0, center + rng.uniform(-45.0, 75.0)),
            max(0.0, shift_duration_s - 1.0),
        )
        src, dst = rng.choice(reachable_pairs)
        run_type = rng.choice(run_types)
        stat_run = ScheduledRun(
            run_type=run_type,
            origin=src,
            destination=dst,
            frequency_minutes=60.0,
            label=f"STAT {run_type}",
        )
        nurse_nodes, nurse_pixels, nurse_dist = shortest_path_with_pixels(
            full_graph, unit, src, dst
        )
        nurse_route = _PlannedRoute(
            run=stat_run,
            agent_role="nurse",
            node_path=nurse_nodes,
            pixel_path=nurse_pixels,
            distance_m=nurse_dist,
        )
        robot_route = _plan_robot_assisted_route(
            unit, full_graph, robot_graph, stat_run, num_robots
        )
        speed_factor = rng.uniform(0.85, 1.15)
        delay_s = rng.uniform(10.0, max_delay) if rng.random() < delay_prob else 0.0
        stat_requests.append(
            _TripRequest(
                arrival_time_s=arrival_s,
                nurse_route=nurse_route,
                robot_route=robot_route,
                speed_factor=speed_factor,
                delay_s=delay_s,
                is_stat=True,
            )
        )

    return stat_requests


def _shortest_deadhead_m(graph: nx.Graph, src: str | None, dst: str) -> float:
    if src is None or src == dst or src not in graph or dst not in graph:
        return 0.0
    try:
        return float(nx.shortest_path_length(graph, source=src, target=dst, weight="weight"))
    except nx.NetworkXNoPath:
        return float("inf")


def _run_single_mode(
    requests: list[_TripRequest],
    routing_graph: nx.Graph,
    mode: Literal["nurse_only", "robot_assisted"],
    shift_duration_s: float,
    num_robots: int,
    nurse_speed_mps: float,
    robot_speed_mps: float,
) -> ShiftSummary:
    env = simpy.Environment()
    events: list[TripEvent] = []

    nurse_pool = simpy.FilterStore(env, capacity=DEFAULT_NUM_NURSES)
    nurse_loc: dict[str, str | None] = {}
    nurse_counts: dict[str, int] = {}
    for i in range(DEFAULT_NUM_NURSES):
        nid = f"nurse-{i + 1}"
        nurse_pool.put(nid)
        nurse_loc[nid] = None
        nurse_counts[nid] = 0

    robot_pool: simpy.FilterStore | None = None
    robot_loc: dict[str, str | None] = {}
    robot_counts: dict[str, int] = {}
    if mode == "robot_assisted" and num_robots > 0:
        robot_pool = simpy.FilterStore(env, capacity=num_robots)
        for i in range(num_robots):
            rid = f"robot-{i + 1}"
            robot_pool.put(rid)
            robot_loc[rid] = None
            robot_counts[rid] = 0

    def execute_trip(req: _TripRequest):
        route = req.robot_route if mode == "robot_assisted" else req.nurse_route
        use_robot = route.agent_role == "robot" and robot_pool is not None
        pool = robot_pool if use_robot else nurse_pool
        loc_map = robot_loc if use_robot else nurse_loc
        count_map = robot_counts if use_robot else nurse_counts
        role: Literal["nurse", "robot"] = "robot" if use_robot else "nurse"
        base_speed = robot_speed_mps if use_robot else nurse_speed_mps
        effective_speed = max(0.2, base_speed * req.speed_factor)

        requested_at = env.now
        if pool.items:
            origin_node = route.node_path[0]
            best_id = min(
                pool.items,
                key=lambda aid: (
                    _shortest_deadhead_m(routing_graph, loc_map.get(aid), origin_node),
                    count_map.get(aid, 0),
                    aid,
                ),
            )
            agent_id = yield pool.get(lambda aid: aid == best_id)
        else:
            agent_id = yield pool.get()

        start_time_s = env.now
        wait_time_s = start_time_s - requested_at
        travel_time_s = (
            (route.distance_m / effective_speed) + req.delay_s
            if route.distance_m > 0
            else req.delay_s
        )
        yield env.timeout(travel_time_s)
        end_time_s = env.now

        loc_map[agent_id] = route.node_path[-1]
        count_map[agent_id] = count_map.get(agent_id, 0) + 1
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

    def dispatcher():
        for req in requests:
            delay = req.arrival_time_s - env.now
            if delay > 0:
                yield env.timeout(delay)
            env.process(execute_trip(req))

    env.process(dispatcher())
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
    scenario_mode: ScenarioMode = "routine",
    seed: int = 42,
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
    shift_duration_s = shift_hours * 3600.0

    requests = _generate_trip_requests(
        unit=unit,
        full_graph=full_graph,
        robot_graph=robot_graph,
        schedule=schedule,
        shift_duration_s=shift_duration_s,
        num_robots=num_robots,
        scenario_mode=scenario_mode,
        seed=seed,
    )

    nurse_only = _run_single_mode(
        requests=requests,
        routing_graph=full_graph,
        mode="nurse_only",
        shift_duration_s=shift_duration_s,
        num_robots=0,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
    )
    robot_assisted = _run_single_mode(
        requests=requests,
        routing_graph=robot_graph,
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
    stat_trips = sum(1 for req in requests if req.is_stat)

    robot_waits = [e.wait_time_s for e in robot_assisted.events]
    nurse_waits = [e.wait_time_s for e in nurse_only.events]

    assumptions: dict[str, float | str] = {
        "shift_hours": shift_hours,
        "num_nurses": DEFAULT_NUM_NURSES,
        "num_robots": num_robots,
        "nurse_speed_mps": nurse_speed_mps,
        "robot_speed_mps": robot_speed_mps,
        "meters_per_pixel": unit.meters_per_pixel,
        "scenario_mode": scenario_mode,
        "seed": seed,
        "stat_trips": stat_trips,
        "autonomized_trips": autonomized_trips,
        "human_only_trips": human_only_trips,
        "max_wait_s": round(max(robot_waits, default=0.0), 3),
        "p95_wait_s": round(_percentile(robot_waits, 95.0), 3),
        "nurse_only_max_wait_s": round(max(nurse_waits, default=0.0), 3),
        "nurse_only_p95_wait_s": round(_percentile(nurse_waits, 95.0), 3),
        "trip_model": (
            "Autonomy-aware shortest path via room centroids and doors with nearest-free-robot "
            "dispatch; human_only rooms use nearest autonomous/assisted handoff nodes, "
            "and robot-unreachable runs fall back to nurses."
        ),
    }

    return SimulationComparison(
        assumptions=assumptions,
        nurse_only=nurse_only,
        robot_assisted=robot_assisted,
        nurse_minutes_saved=nurse_minutes_saved,
    )
