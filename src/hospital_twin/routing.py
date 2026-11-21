"""Route graph construction and shortest-path lookup over a UnitMap."""

from __future__ import annotations

import logging
import math

import networkx as nx

from hospital_twin.schemas import Point, Room, UnitMap


def room_centroid(room: Room) -> Point:
    """Return the room's centroid, computing it from its polygon if omitted."""
    if room.centroid is not None:
        return room.centroid
    pts = room.polygon
    if not pts:
        raise ValueError(f"Room '{room.id}' has neither centroid nor polygon points.")
    if len(pts) < 3:
        return Point(
            x=sum(p.x for p in pts) / len(pts),
            y=sum(p.y for p in pts) / len(pts),
        )

    # Shoelace formula for polygon centroid; falls back to vertex mean if degenerate.
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        cross = p0.x * p1.y - p1.x * p0.y
        area2 += cross
        cx += (p0.x + p1.x) * cross
        cy += (p0.y + p1.y) * cross

    if abs(area2) < 1e-9:
        return Point(
            x=sum(p.x for p in pts) / n,
            y=sum(p.y for p in pts) / n,
        )
    factor = 1.0 / (3.0 * area2)
    return Point(x=cx * factor, y=cy * factor)


def _pixel_distance(a: Point, b: Point) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


DEFAULT_ROBOT_CLEARANCE_M = 0.90


def build_route_graph(
    unit: UnitMap,
    robot_only: bool = False,
    robot_clearance_m: float = DEFAULT_ROBOT_CLEARANCE_M,
) -> nx.Graph:
    """Build a NetworkX graph where rooms are nodes and doors are weighted edges.

    Edge weights are Euclidean pixel distances (room centroid -> door -> room centroid)
    multiplied by ``unit.meters_per_pixel``.

    When ``robot_only=True``, rooms with ``autonomy_tier == "human_only"`` and
    doors with ``robot_passable == False`` or ``width_m < robot_clearance_m``
    are excluded.
    """
    graph = nx.Graph()
    # People can always get there somehow; robots cannot. Only the walking graph estimates
    # a route across a missing doorway (see shortest_path_with_pixels).
    graph.graph["estimate_missing_routes"] = not robot_only
    centroids: dict[str, Point] = {}

    for room in unit.rooms:
        if robot_only and room.autonomy_tier == "human_only":
            continue
        c = room_centroid(room)
        centroids[room.id] = c
        graph.add_node(
            room.id,
            label=room.label,
            type=room.type,
            autonomy_tier=room.autonomy_tier,
            pos=(c.x, c.y),
        )

    for door in unit.doors:
        if robot_only and (
            door.width_m < robot_clearance_m
            or (not door.robot_passable and robot_clearance_m >= DEFAULT_ROBOT_CLEARANCE_M)
        ):
            continue
        u, v = door.rooms_connected
        if u == v or u not in centroids or v not in centroids:
            continue
        px_dist = _pixel_distance(centroids[u], door.position) + _pixel_distance(
            door.position, centroids[v]
        )
        dist_m = px_dist * unit.meters_per_pixel

        # Keep the shortest door if multiple doors connect the same pair of rooms.
        if graph.has_edge(u, v) and graph[u][v]["weight"] <= dist_m:
            continue
        graph.add_edge(
            u,
            v,
            weight=dist_m,
            door_id=door.id,
            door_pos=(door.position.x, door.position.y),
        )

    return graph


def find_robot_handoff_node(unit: UnitMap, target_room_id: str) -> str | None:
    """Find the nearest autonomous/assisted node connected to ``target_room_id``.

    If ``target_room_id`` is already ``autonomous`` or ``assisted``, returns
    ``target_room_id``. If ``target_room_id`` is ``human_only``, returns the
    nearest reachable ``autonomous`` or ``assisted`` room ID, or ``None`` if none.
    """
    room_by_id = {r.id: r for r in unit.rooms}
    target = room_by_id.get(target_room_id)
    if target is None:
        return None
    if target.autonomy_tier != "human_only":
        return target_room_id

    full_graph = build_route_graph(unit, robot_only=False)
    if target_room_id not in full_graph:
        return None

    distances = nx.single_source_dijkstra_path_length(
        full_graph, target_room_id, weight="weight"
    )
    candidates = [
        (dist, nid)
        for nid, dist in distances.items()
        if nid != target_room_id
        and nid in room_by_id
        and room_by_id[nid].autonomy_tier in ("autonomous", "assisted")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def shortest_path_with_pixels(
    G: nx.Graph,
    unit: UnitMap,
    src: str,
    dst: str,
) -> tuple[list[str], list[Point], float]:
    """Compute the shortest path between two rooms and its pixel polyline.

    Returns ``(node_path, pixel_path, distance_m)``.
    """
    if src not in G:
        raise ValueError(f"Source room '{src}' not found in route graph.")
    if dst not in G:
        raise ValueError(f"Destination room '{dst}' not found in route graph.")

    centroids = {r.id: room_centroid(r) for r in unit.rooms}

    if src == dst:
        return [src], [centroids[src]], 0.0

    try:
        node_path: list[str] = nx.shortest_path(G, source=src, target=dst, weight="weight")
    except nx.NetworkXNoPath as exc:
        if not G.graph.get("estimate_missing_routes"):
            raise ValueError(f"No route between '{src}' and '{dst}'.") from exc
        # Disconnected rooms usually mean a doorway the extractor missed. Estimate the trip
        # as a straight line with a detour factor instead of failing the whole shift.
        a, b = centroids[src], centroids[dst]
        mpp = getattr(unit, "meters_per_pixel", 0.05) or 0.05
        dist_m = math.hypot(b.x - a.x, b.y - a.y) * mpp * 1.4
        logging.getLogger(__name__).warning("No route between %s and %s; using estimated path.", src, dst)
        return [src, dst], [a, b], dist_m

    pixel_path: list[Point] = [centroids[node_path[0]]]
    total_m = 0.0
    for u, v in zip(node_path, node_path[1:]):
        edge = G[u][v]
        door_x, door_y = edge["door_pos"]
        pixel_path.append(Point(x=door_x, y=door_y))
        pixel_path.append(centroids[v])
        total_m += float(edge["weight"])

    return node_path, pixel_path, total_m
