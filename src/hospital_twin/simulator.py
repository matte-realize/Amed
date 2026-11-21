"""SimPy discrete-event shift simulation comparing nurse-only vs. robot-assisted runs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal

import networkx as nx
import simpy

from hospital_twin.routing import (
    DEFAULT_ROBOT_CLEARANCE_M,
    build_route_graph,
    find_robot_handoff_node,
    shortest_path_with_pixels,
)
from hospital_twin.schemas import (
    CustodyLogEntry,
    Point,
    RunType,
    ScheduledRun,
    ShiftSummary,
    SimulationComparison,
    TripEvent,
    UnitMap,
)

DEFAULT_NUM_NURSES = 2
DEFAULT_ROBOT_SPEED_MPS = 0.76  # Aethon TUG specification: 76 cm/s
DEFAULT_ROBOT_LENGTH_M = 0.965  # Aethon TUG specification: 96.5 cm
DEFAULT_HANDOFF_SECONDS = 20.0  # 20s badge-and-unlock handoff at every load and delivery
UNCOLLECTED_FLAG_SECONDS = 600.0  # 10-minute uncollected delivery exception threshold
POD_COMPARTMENTS = 3  # pharmacy, specimens, supplies
ScenarioMode = Literal[
    "routine",
    "chaotic",
    "surge",
    "robot_failure",
    "pharmacy_delay",
    "staffing_gap",
    "crash_cart_missing",
]


def _format_shift_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


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
            effective_robot_route = robot_route
            if scenario_mode == "routine":
                arrival_s = t
                speed_factor = 1.0
                delay_s = 0.0
            elif scenario_mode == "robot_failure":
                arrival_s = t
                speed_factor = rng.uniform(0.90, 1.10)
                delay_s = 0.0
                in_outage = 0.40 * shift_duration_s <= arrival_s <= 0.65 * shift_duration_s
                if in_outage and (num_robots <= 1 or rng.random() < 0.5):
                    # Safe degradation: robot blocked/offline, falls back to nurse
                    effective_robot_route = nurse_route
            elif scenario_mode == "pharmacy_delay":
                arrival_s = t
                speed_factor = rng.uniform(0.88, 1.12)
                delay_s = (
                    rng.uniform(60.0, 180.0)
                    if run.run_type == "pharmacy_pickup"
                    else 0.0
                )
            elif scenario_mode == "staffing_gap":
                arrival_s = t
                speed_factor = rng.uniform(0.85, 1.05)
                delay_s = rng.uniform(15.0, 45.0) if (0.35 * shift_duration_s <= t <= 0.65 * shift_duration_s) else 0.0
            elif scenario_mode == "crash_cart_missing":
                arrival_s = t
                speed_factor = rng.uniform(0.85, 1.05)
                delay_s = (
                    rng.uniform(90.0, 210.0)
                    if run.run_type == "supply_restock"
                    else 0.0
                )
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
                    robot_route=effective_robot_route,
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


def _batch_robot_requests(
    requests: list[_TripRequest],
    mode: Literal["nurse_only", "robot_assisted"],
    num_robots: int,
) -> list[list[_TripRequest]]:
    """Group simultaneous/near-simultaneous distinct-compartment robot runs into 3-compartment pod batches."""
    if mode != "robot_assisted" or num_robots <= 0:
        return [[req] for req in requests]

    batches: list[list[_TripRequest]] = []
    i = 0
    n = len(requests)
    while i < n:
        first = requests[i]
        if first.robot_route.agent_role != "robot":
            batches.append([first])
            i += 1
            continue

        # Only batch when there are multiple distinct-compartment requests arriving together
        # and fewer robots than concurrent requests (e.g., 1 robot with a 3-compartment pod).
        batch = [first]
        seen_types: set[RunType] = {first.robot_route.run.run_type}
        j = i + 1
        window_end = first.arrival_time_s + 60.0
        while (
            j < n
            and len(batch) < POD_COMPARTMENTS
            and requests[j].arrival_time_s <= window_end
            and requests[j].robot_route.agent_role == "robot"
            and requests[j].robot_route.run.run_type not in seen_types
        ):
            # If we have enough free robots to serve all concurrent runs immediately, keep unbatched
            # so multi-robot parallelism is preserved; otherwise batch into the 3-compartment pod.
            if num_robots >= POD_COMPARTMENTS:
                break
            batch.append(requests[j])
            seen_types.add(requests[j].robot_route.run.run_type)
            j += 1

        batches.append(batch)
        i = j
    return batches


def _run_single_mode(
    requests: list[_TripRequest],
    routing_graph: nx.Graph,
    room_labels: dict[str, str],
    mode: Literal["nurse_only", "robot_assisted"],
    shift_duration_s: float,
    num_robots: int,
    nurse_speed_mps: float,
    robot_speed_mps: float,
    handoff_seconds: float = DEFAULT_HANDOFF_SECONDS,
    uncollected_flag_s: float = UNCOLLECTED_FLAG_SECONDS,
) -> ShiftSummary:
    env = simpy.Environment()
    events: list[TripEvent] = []
    custody_log: list[CustodyLogEntry] = []
    order_seq = 0
    batch_seq = 0

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

    def _room_name(room_id: str) -> str:
        return room_labels.get(room_id, room_id)

    def execute_batch(batch: list[_TripRequest]):
        nonlocal order_seq, batch_seq
        lead = batch[0]
        lead_route = lead.robot_route if mode == "robot_assisted" else lead.nurse_route
        use_robot = lead_route.agent_role == "robot" and robot_pool is not None
        pool = robot_pool if use_robot else nurse_pool
        loc_map = robot_loc if use_robot else nurse_loc
        count_map = robot_counts if use_robot else nurse_counts
        role: Literal["nurse", "robot"] = "robot" if use_robot else "nurse"
        base_speed = robot_speed_mps if use_robot else nurse_speed_mps

        requested_at = env.now
        if pool.items:
            origin_node = lead_route.node_path[0]
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

        batch_id: str | None = None
        if use_robot and len(batch) > 1:
            batch_seq += 1
            batch_id = f"POD-{batch_seq:03d}"

        who_label = f"{agent_id} (TUG+Pod)" if use_robot else agent_id

        for idx, req in enumerate(batch):
            order_seq += 1
            order_no = f"ORD-{order_seq:03d}"
            route = req.robot_route if mode == "robot_assisted" else req.nurse_route
            effective_speed = max(0.2, base_speed * req.speed_factor)

            start_time_s = env.now
            wait_time_s = max(0.0, start_time_s - req.arrival_time_s)

            # Badge-and-unlock handoff at load (once for batch lead, or per unbatched trip)
            load_handoff = handoff_seconds if (use_robot and idx == 0) else 0.0
            if use_robot:
                custody_log.append(
                    CustodyLogEntry(
                        order_number=order_no,
                        who=who_label,
                        where=_room_name(route.node_path[0]),
                        when_s=round(start_time_s, 3),
                        when_formatted=_format_shift_time(start_time_s),
                        action="LOAD",
                        run_type=route.run.run_type,
                    )
                )

            transit_s = (
                (route.distance_m / effective_speed) + req.delay_s
                if route.distance_m > 0
                else req.delay_s
            )
            delivery_handoff = handoff_seconds if use_robot else 0.0
            step_duration_s = load_handoff + transit_s + delivery_handoff
            yield env.timeout(step_duration_s)
            end_time_s = env.now

            if use_robot:
                uncollected = (end_time_s - req.arrival_time_s) > uncollected_flag_s
                custody_log.append(
                    CustodyLogEntry(
                        order_number=order_no,
                        who=who_label,
                        where=_room_name(route.node_path[-1]),
                        when_s=round(end_time_s, 3),
                        when_formatted=_format_shift_time(end_time_s),
                        action="FLAG_UNCOLLECTED" if uncollected else "DELIVER",
                        run_type=route.run.run_type,
                        flagged_uncollected=uncollected,
                    )
                )

            loc_map[agent_id] = route.node_path[-1]
            count_map[agent_id] = count_map.get(agent_id, 0) + 1

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
                    batch_id=batch_id,
                )
            )

        yield pool.put(agent_id)

    batches = _batch_robot_requests(requests, mode, num_robots)

    def dispatcher():
        for batch in batches:
            delay = batch[0].arrival_time_s - env.now
            if delay > 0:
                yield env.timeout(delay)
            env.process(execute_batch(batch))

    env.process(dispatcher())
    env.run()

    events.sort(key=lambda e: (e.start_time_s, e.end_time_s, e.agent_id))
    custody_log.sort(key=lambda c: (c.when_s, c.order_number))

    nurse_minutes_walked = sum(
        (e.end_time_s - e.start_time_s) / 60.0 for e in events if e.agent_role == "nurse"
    )
    robot_minutes_traveled = sum(
        (e.end_time_s - e.start_time_s) / 60.0 for e in events if e.agent_role == "robot"
    )
    avg_wait_s = sum(e.wait_time_s for e in events) / len(events) if events else 0.0
    flagged_exceptions = sum(1 for c in custody_log if c.flagged_uncollected)

    return ShiftSummary(
        mode=mode,
        num_robots=num_robots if mode == "robot_assisted" else 0,
        shift_duration_s=round(shift_duration_s, 3),
        nurse_minutes_walked=round(nurse_minutes_walked, 3),
        robot_minutes_traveled=round(robot_minutes_traveled, 3),
        avg_delivery_wait_s=round(avg_wait_s, 3),
        total_trips=len(events),
        events=events,
        custody_log=custody_log,
        flagged_exceptions=flagged_exceptions,
    )


def simulate_shift(
    unit: UnitMap,
    schedule: list[ScheduledRun],
    shift_hours: float = 8.0,
    num_robots: int = 1,
    nurse_speed_mps: float = 1.2,
    robot_speed_mps: float = DEFAULT_ROBOT_SPEED_MPS,
    handoff_seconds: float = DEFAULT_HANDOFF_SECONDS,
    robot_clearance_m: float = DEFAULT_ROBOT_CLEARANCE_M,
    scenario_mode: ScenarioMode = "routine",
    seed: int = 42,
    ems_preposition: bool = False,
) -> SimulationComparison:
    """Simulate a hospital shift in nurse-only and robot-assisted modes using SimPy."""
    if shift_hours <= 0:
        raise ValueError("shift_hours must be positive.")
    if num_robots < 0:
        raise ValueError("num_robots must be non-negative.")
    if nurse_speed_mps <= 0 or robot_speed_mps <= 0:
        raise ValueError("nurse_speed_mps and robot_speed_mps must be positive.")

    full_graph = build_route_graph(unit, robot_only=False)
    robot_graph = build_route_graph(unit, robot_only=True, robot_clearance_m=robot_clearance_m)
    room_labels = {room.id: room.label for room in unit.rooms}
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
    if ems_preposition:
        # Advance EMS signals pre-stage carts/meds before arrival, cutting STAT surge delays by 70%
        requests = [
            _TripRequest(
                arrival_time_s=req.arrival_time_s,
                nurse_route=req.nurse_route,
                robot_route=req.robot_route,
                speed_factor=req.speed_factor,
                delay_s=round(req.delay_s * 0.3, 3) if req.is_stat else req.delay_s,
                is_stat=req.is_stat,
            )
            for req in requests
        ]

    nurse_only = _run_single_mode(
        requests=requests,
        routing_graph=full_graph,
        room_labels=room_labels,
        mode="nurse_only",
        shift_duration_s=shift_duration_s,
        num_robots=0,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
        handoff_seconds=handoff_seconds,
    )
    robot_assisted = _run_single_mode(
        requests=requests,
        routing_graph=robot_graph,
        room_labels=room_labels,
        mode="robot_assisted",
        shift_duration_s=shift_duration_s,
        num_robots=num_robots,
        nurse_speed_mps=nurse_speed_mps,
        robot_speed_mps=robot_speed_mps,
        handoff_seconds=handoff_seconds,
    )

    nurse_minutes_saved = round(
        nurse_only.nurse_minutes_walked - robot_assisted.nurse_minutes_walked, 3
    )
    autonomized_trips = sum(1 for e in robot_assisted.events if e.agent_role == "robot")
    human_only_trips = sum(1 for e in robot_assisted.events if e.agent_role == "nurse")
    batched_trips = sum(1 for e in robot_assisted.events if e.batch_id is not None)
    stat_trips = sum(1 for req in requests if req.is_stat)

    robot_waits = [e.wait_time_s for e in robot_assisted.events]
    nurse_waits = [e.wait_time_s for e in nurse_only.events]

    assumptions: dict[str, float | str] = {
        "shift_hours": shift_hours,
        "num_nurses": DEFAULT_NUM_NURSES,
        "num_robots": num_robots,
        "nurse_speed_mps": nurse_speed_mps,
        "robot_speed_mps": robot_speed_mps,
        "robot_length_m": DEFAULT_ROBOT_LENGTH_M,
        "robot_clearance_m": robot_clearance_m,
        "handoff_seconds": handoff_seconds,
        "pod_compartments": POD_COMPARTMENTS,
        "uncollected_flag_seconds": UNCOLLECTED_FLAG_SECONDS,
        "meters_per_pixel": unit.meters_per_pixel,
        "scenario_mode": scenario_mode,
        "seed": seed,
        "stat_trips": stat_trips,
        "autonomized_trips": autonomized_trips,
        "human_only_trips": human_only_trips,
        "batched_trips": batched_trips,
        "flagged_exceptions": robot_assisted.flagged_exceptions,
        "max_wait_s": round(max(robot_waits, default=0.0), 3),
        "p95_wait_s": round(_percentile(robot_waits, 95.0), 3),
        "nurse_only_max_wait_s": round(max(nurse_waits, default=0.0), 3),
        "nurse_only_p95_wait_s": round(_percentile(nurse_waits, 95.0), 3),
        "unit_source": "VA Medical/Surgical Inpatient Units & ICU Design Guide (Nov 2011); VA Space Planning Criteria Ch. 100 (17-33 beds)",
        "robot_source": "Aethon TUG specifications (76 cm/s, 96.5 cm long)",
        "demand_sanity_check": "Aethon via RIA: ~10,000 med orders/week at 200-bed hospital (plausibility check only)",
        "trip_model": (
            "Aethon TUG (0.76 m/s) + 3-compartment custodial courier pod (pharmacy, specimens, supplies) "
            "with 20s badge-and-unlock handoffs, door clearance fit check (>=0.90 m), and 10-min uncollected flagging."
        ),
    }

    return SimulationComparison(
        assumptions=assumptions,
        nurse_only=nurse_only,
        robot_assisted=robot_assisted,
        nurse_minutes_saved=nurse_minutes_saved,
    )
