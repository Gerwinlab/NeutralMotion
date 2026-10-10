from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import _is_reorderable_cz, format_node_line, op_node_signature
from .grid import Qubit

#TODO: Make sure to separate two-qubit pulse based on rydberg radius so there is no conflicts.

MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent


def _is_two_qubit_operation(node):
    return len(node.qargs) == 2 and node.op.name not in {"swap", "barrier"}


def _max_grid_spans(max_dimension: list[int]) -> tuple[int, int]:
    """Convert AOD atom dimensions into max grid-index spans on even-even lattice sites."""
    max_rows, max_cols = max_dimension
    return 2 * (max_rows - 1), 2 * (max_cols - 1)


def _is_valid_aod_position(position: tuple[int, int]) -> bool:
    """Return True when a position is not an even-even SLM trap site."""
    return not (position[0] % 2 == 0 and position[1] % 2 == 0)


def _fits_same_aod(
    positions: list[tuple[int, int]],
    candidate: tuple[int, int],
    max_row_span: int,
    max_col_span: int,
) -> bool:
    """Return True when `candidate` can be added while all points stay in one AOD window."""
    if not positions:
        return True
    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    cand_r, cand_c = candidate
    row_span = max(max(rows), cand_r) - min(min(rows), cand_r)
    col_span = max(max(cols), cand_c) - min(min(cols), cand_c)
    return row_span <= max_row_span and col_span <= max_col_span


def _movement_vector(
    moving_pos: tuple[int, int],
    other_pos: tuple[int, int],
) -> tuple[int, int]:
    """Vector for moving `moving_pos` toward the partner atom at `other_pos`."""
    return other_pos[0] - moving_pos[0], other_pos[1] - moving_pos[1]


def _vector_alignment_score(
    group_vectors: list[tuple[int, int]],
    candidate_vector: tuple[int, int],
) -> float:
    """Higher is better: parallel direction and similar magnitude to existing group vectors."""
    if not group_vectors:
        return 0.0

    cand_x, cand_y = candidate_vector
    cand_mag = math.hypot(cand_x, cand_y)
    if cand_mag == 0:
        return float("-inf")

    score = 0.0
    for vec_x, vec_y in group_vectors:
        vec_mag = math.hypot(vec_x, vec_y)
        if vec_mag == 0:
            continue
        cosine = (cand_x * vec_x + cand_y * vec_y) / (cand_mag * vec_mag)
        mag_ratio = min(cand_mag, vec_mag) / max(cand_mag, vec_mag)
        score += cosine * mag_ratio
    return score / len(group_vectors)


def _pair_alignment_score(
    reference_vector: tuple[int, int],
    candidate_vector: tuple[int, int],
) -> float:
    """Alignment between two vectors using cosine similarity scaled by magnitude ratio."""
    ref_x, ref_y = reference_vector
    cand_x, cand_y = candidate_vector
    ref_mag = math.hypot(ref_x, ref_y)
    cand_mag = math.hypot(cand_x, cand_y)
    if ref_mag == 0 or cand_mag == 0:
        return float("-inf")
    cosine = (cand_x * ref_x + cand_y * ref_y) / (cand_mag * ref_mag)
    mag_ratio = min(cand_mag, ref_mag) / max(cand_mag, ref_mag)
    return cosine * mag_ratio


def _sort_group_by_alignment(vectors: list[tuple[int, int]]) -> list[int]:
    """Return vector indices: smallest magnitude first, then greedy best pair alignment."""
    if not vectors:
        return []

    remaining = set(range(len(vectors)))
    start_idx = min(remaining, key=lambda idx: (math.hypot(*vectors[idx]), idx))
    order = [start_idx]
    remaining.remove(start_idx)

    while remaining:
        prev_vec = vectors[order[-1]]
        next_idx = max(
            remaining,
            key=lambda idx: (_pair_alignment_score(prev_vec, vectors[idx]), -idx),
        )
        order.append(next_idx)
        remaining.remove(next_idx)

    return order


_MAX_EXACT_DISPLACEMENT_TOUR = 24


@dataclass(frozen=True)
class _LayeredTourGraph:
    """Symmetric TSP instance for one AOD load that stays loaded across layers.

    Node 0 is the load terminal at the current rigid offset; the last node is
    the unload terminal at home (0, 0). Nodes in between are the distinct
    ``cities`` in order, each a (layer, displacement) pair. Terminals count as
    layers -1 and ``layer_count``, closing a ring of layers with one free
    unload-to-load edge. Only edges within a layer or between consecutive
    layers exist; every layer crossing pays ``penalty``. Missing edges are
    ``math.inf`` (an integer solver such as Concorde needs a finite stand-in).
    """

    cities: tuple[tuple[int, tuple[int, int]], ...]
    node_layers: tuple[int, ...]
    weights: tuple[tuple[float, ...], ...]
    penalty: float


def _layered_tour_graph(cities, config, start_offset, *, sides=False):
    """Build the penalized layer-ring graph for labelled (layer, displacement) cities.

    A Hamiltonian cycle crosses every cut an even number of times. With the
    unload-to-load edge, each of the layer_count + 1 boundaries is crossed at
    least once; without it, at least twice. Any tour that splits a layer or
    returns to an earlier one therefore pays two or more extra penalties. The
    penalty exceeds a complete feasible tour's movement, so such tours always
    lose: an exact solver returns a layer-ordered tour, and the movement cost
    is the tour cost minus (layer_count + 1) penalties.

    Edges are priced between target centers, or with ``sides`` by
    ``_side_movement``. Either way there is one node per city and the matrix
    stays symmetric.
    """
    layer_count = max(layer for layer, _ in cities) + 1
    nodes = [(-1, start_offset), *cities, (layer_count, (0, 0))]
    end = len(nodes) - 1
    movement = (_side_movement if sides else _center_movement)(nodes, config)

    ordered = [0, *sorted(range(1, end), key=lambda node: nodes[node][0]), end]
    bound = sum(movement(a, b) for a, b in zip(ordered, ordered[1:]))
    penalty = bound + max(bound, 1e-12)
    weights = tuple(
        tuple(
            0.0 if i == j or {i, j} == {0, end}
            else math.inf if abs(a - b) > 1
            else movement(i, j) + penalty * (a != b)
            for j, (b, _) in enumerate(nodes)
        )
        for i, (a, _) in enumerate(nodes)
    )
    return _LayeredTourGraph(
        tuple(cities), tuple(layer for layer, _ in nodes), weights, penalty,
    )


def _center_movement(nodes, config):
    """Motion between the target centers of two graph nodes."""
    def movement(i, j):
        (_, start), (_, end) = nodes[i], nodes[j]
        return _step_seconds((end[0] - start[0], end[1] - start[1]), config)
    return movement


def _side_movement(nodes, config):
    """Cheapest parity-routed motion between any interaction sides of two nodes.

    A city is visited from one of four cardinal sites around its even-even
    target. The load terminal starts at its offset, or at home with any
    transfer direction when that offset is (0, 0); the unload terminal is
    reached by routing to any home-adjacent site and transferring down. The
    minimum is symmetric because route costs are, so the graph keeps one node
    per city. Leaving and arriving sides may differ, so this prices order
    only; ``_optimize_visit_sides`` then chooses consistent sides exactly.
    """
    end = len(nodes) - 1
    transfers = [(_step_seconds(step, config), step) for step in _CARDINAL]
    start = nodes[0][1]
    entries = transfers if start == (0, 0) else [(0.0, start)]
    sites = [[(0.0, (r + dr, c + dc)) for dr, dc in _CARDINAL] for _, (r, c) in nodes]

    def route(a, b):
        return _route(a, b, config)[0]

    @lru_cache(maxsize=None)
    def movement(i, j):
        if i > j:
            return movement(j, i)
        first = entries if i == 0 else sites[i]
        second = transfers if j == end else sites[j]
        return min(a + route(p, q) + b for a, p in first for b, q in second)
    return movement


def _route(start, end, config):
    """Parity-routed (motion seconds, steps) from ``start`` to ``end``, cached.

    Highways depend only on coordinate parity, so a route is fixed by the
    start's parity class and the displacement; translating both endpoints by
    an even vector leaves it unchanged. The cache is shared across groups.
    """
    return _route_steps(
        (start[0] % 2, start[1] % 2), (end[0] - start[0], end[1] - start[1]),
        config.get("_motion_parameters") or _motion_parameters(config),
        config.get("movement_shape", "trapezoidal"),
    )


@lru_cache(maxsize=1 << 16)
def _route_steps(parity, delta, motion_parameters, movement_shape):
    config = {"_motion_parameters": motion_parameters, "movement_shape": movement_shape}
    steps = tuple(parity_route_moves(parity, (parity[0] + delta[0], parity[1] + delta[1]), config))
    return sum(_step_seconds(step, config) for step in steps), steps


def _held_karp_tour(weights, node_layers=None):
    """Exact minimum Hamiltonian cycle through node 0 of a symmetric matrix.

    Missing edges are ``math.inf`` and are never traversed. Without
    ``node_layers`` this is plain Held-Karp over all subsets. With the labels of
    a ``_layered_tour_graph``, only paths that finish a layer before entering
    the next are expanded; the penalty already makes every other tour more
    expensive, so the optimum is unchanged while the subsets shrink to layer
    prefixes.
    """
    count = len(weights)
    if count == 1:
        return [0]
    full_mask = (1 << count) - 1
    layer_masks = defaultdict(int)
    for node, layer in enumerate(node_layers or ()):
        layer_masks[layer] |= 1 << node

    # Costs within float noise of each other (summation order, penalty
    # offsets) tie; ties prefer the lower predecessor index, deterministically.
    tolerance = 1e-12 * max(1.0, sum(
        max((w for w in row if math.isfinite(w)), default=0.0) for row in weights
    ))
    # (mask, last) -> (cost of a node-0-to-last path visiting mask, previous node)
    best = {(1, 0): (0.0, -1)}
    frontier = [(1, 0)]
    for _ in range(count - 1):
        expanded = {}
        for mask, last in frontier:
            cost = best[mask, last][0]
            if node_layers is None:
                candidates = full_mask & ~mask
            else:
                layer = node_layers[last]
                candidates = layer_masks[layer] & ~mask or layer_masks.get(layer + 1, 0)
            while candidates:
                bit = candidates & -candidates
                candidates ^= bit
                node = bit.bit_length() - 1
                step = weights[last][node]
                if not math.isfinite(step):
                    continue
                key = (mask | bit, node)
                total = cost + step
                previous = expanded.get(key)
                if previous is None or total < previous[0] - tolerance or (
                    total <= previous[0] + tolerance and last < previous[1]
                ):
                    expanded[key] = (total, last)
        best.update(expanded)
        frontier = list(expanded)

    optimum, last = math.inf, -1
    for mask, node in frontier:
        if mask == full_mask:
            total = best[full_mask, node][0] + weights[node][0]
            if last == -1 or total < optimum - tolerance or (
                total <= optimum + tolerance and node < last
            ):
                optimum, last = total, node
    if not math.isfinite(optimum):
        raise ValueError("No Hamiltonian tour exists in the displacement graph.")
    tour, mask = [], full_mask
    while last != -1:
        tour.append(last)
        last, mask = best[mask, last][1], mask ^ (1 << last)
    return tour[::-1]


def _validated_layer_order(tour, graph):
    """Return city indices of a solver's tour after checking it is faithful.

    Exact solvers cannot violate the layer ring, but approximate ones can. The
    tour must be a Hamiltonian cycle over existing edges that closes through
    the unload-to-load edge and, oriented from loading, never revisits a layer.
    """
    tour = list(tour)
    count = len(graph.weights)
    end = count - 1
    if len(tour) != count or set(tour) != set(range(count)):
        raise ValueError("TSP solver did not return a Hamiltonian tour.")
    if any(not math.isfinite(graph.weights[a][b]) for a, b in zip(tour, tour[1:] + tour[:1])):
        raise ValueError("TSP solver tour uses an edge absent from the layer graph.")
    start = tour.index(0)
    cycle = tour[start:] + tour[:start]
    if cycle[1] == end:
        cycle = [0, *reversed(cycle[1:])]
    layers = [graph.node_layers[node] for node in cycle]
    if cycle[-1] != end or layers != sorted(layers):
        raise ValueError("Displacement tour violated layer order.")
    return [node - 1 for node in cycle[1:-1]]


def _held_karp_displacement_order(
    vectors: list[tuple[int, int]], config: dict,
    *, start_offset: tuple[int, int] = (0, 0),
    layer_ids: list[int] | None = None,
    tour_solver=None,
) -> list[int]:
    """Return all vector indices in an exact minimum-cost displacement tour.

    Cities are partner positions relative to each mover's immutable home.
    The path starts at the current rigid-load offset and ends at home (0, 0),
    not at the starting offset of a reused load. Edge weights use the configured
    movement-time model, approximating targets without interaction-side routing.
    Cities are (layer, displacement) pairs on the symmetric penalized graph of
    ``_layered_tour_graph``; without ``layer_ids`` there is a single layer.
    Repeated displacements within a layer expand consecutively in input order.

    Edges are priced between target centers. Physical targets (partner homes)
    are even-even, so the graph is also solved with cheapest-side pricing
    (``_side_movement``); both orders are scored by the exact fixed-order side
    DP and the cheaper is kept, center order on ties (greedy side ordering
    uses center order only). Neither pricing alone is the side-aware optimum,
    but under the side DP the result is never worse than center pricing.

    ``tour_solver`` maps the weight matrix to a Hamiltonian cycle (for example
    a Concorde wrapper); it defaults to exact Held-Karp. Every returned tour is
    validated before use.
    """
    labels = _displacement_layer_labels(vectors, layer_ids)
    if not vectors:
        return []

    indices_by_city = {}
    for index, city in enumerate(zip(labels, vectors)):
        indices_by_city.setdefault(city, []).append(index)
    cities = list(indices_by_city)

    def solve(sides):
        graph = _layered_tour_graph(cities, config, start_offset, sides=sides)
        if tour_solver is None:
            tour = _held_karp_tour(graph.weights, graph.node_layers)
        else:
            tour = tour_solver(graph.weights)
        return [cities[city] for city in _validated_layer_order(tour, graph)]

    def side_motion(order):
        return _optimize_visit_sides(
            [vector for _, vector in order], config,
            start_offset=None if start_offset == (0, 0) else start_offset,
        ).motion_seconds

    order = solve(False)
    # Greedy side ordering never consults the side DP, so it keeps center order.
    # Sides need even-even targets and a load start at home or an AOD site.
    if (config.get("_side_ordering") != "greedy"
            and all(r % 2 == 0 and c % 2 == 0 for r, c in vectors)
            and (start_offset == (0, 0) or _is_valid_aod_position(start_offset))):
        side_order = solve(True)
        if side_order != order and side_motion(side_order) < side_motion(order):
            order = side_order
    return [vector_index for city in order for vector_index in indices_by_city[city]]


def _displacement_layer_labels(vectors, layer_ids):
    """Normalize chronological labels, allowing empty intervening DAG layers."""
    if layer_ids is None:
        return [0] * len(vectors)
    if len(layer_ids) != len(vectors) or any(
        not isinstance(label, int) or isinstance(label, bool) or label < 0 for label in layer_ids
    ):
        raise ValueError("Expected one nonnegative integer layer ID per displacement.")
    ranks = {label: rank for rank, label in enumerate(sorted(set(layer_ids)))}
    return [ranks[label] for label in layer_ids]


def _order_group_displacements(
    vectors: list[tuple[int, int]], config: dict,
    *, start_offset: tuple[int, int] = (0, 0),
    layer_ids: list[int] | None = None,
) -> list[int]:
    """Order targets without crossing layer boundaries, including in fallback."""
    labels = _displacement_layer_labels(vectors, layer_ids)
    unique_count = len(set(zip(labels, vectors)))
    if unique_count <= _MAX_EXACT_DISPLACEMENT_TOUR:
        if layer_ids is not None:
            return _held_karp_displacement_order(
                vectors, config, start_offset=start_offset, layer_ids=layer_ids,
            )
        return _held_karp_displacement_order(vectors, config, start_offset=start_offset)

    warnings.warn(
        f"Group has {unique_count} unique displacements (counted separately per layer); exact Held-Karp "
        f"ordering is limited to {_MAX_EXACT_DISPLACEMENT_TOUR}. Falling "
        "back to alignment ordering.",
        RuntimeWarning,
        stacklevel=2,
    )
    order = []
    for label in sorted(set(labels)):
        buckets = {}
        for i, stage in enumerate(labels):
            if stage == label:
                buckets.setdefault(vectors[i], []).append(i)
        unique = list(buckets)
        local = _sort_group_by_alignment([
            (vector[0] - start_offset[0], vector[1] - start_offset[1]) for vector in unique
        ])
        order.extend(i for index in local for i in buckets[unique[index]])
    return order


@lru_cache(maxsize=8192)
def _profile_seconds(distance: float, velocity: float, acceleration: float) -> float:
    if distance == 0:
        return 0.0
    ramp_distance = velocity * velocity / acceleration
    if distance > ramp_distance:
        return 2 * velocity / acceleration + (distance - ramp_distance) / velocity
    return 2 * math.sqrt(distance / acceleration)


def _step_seconds(step, config):
    spacing, velocity, acceleration = config.get("_motion_parameters") or _motion_parameters(config)
    dr = abs(step[0]) * spacing
    dc = abs(step[1]) * spacing
    movement_shape = config.get("movement_shape", "trapezoidal")
    if movement_shape == "Cubic_Spline":
        return math.sqrt(6 * dr / acceleration) + math.sqrt(6 * dc / acceleration)
    if movement_shape == "trapezoidal":
        return _profile_seconds(math.hypot(dr, dc), velocity, acceleration)
    raise ValueError(f"Unsupported movement_shape: {movement_shape!r}.")


def _motion_parameters(config):
    return (
        config["rydberg_radius"].to("meter").magnitude / 2,
        config["max_velocity"].to("meter/second").magnitude,
        config["max_acceleration"].to("meter/second^2").magnitude,
    )


def _highway_segment(start, end):
    """Check the whole straight segment, including interior SLM sites."""
    if not (_is_valid_aod_position(start) and _is_valid_aod_position(end)):
        return False
    dr, dc = _movement_vector(start, end)
    if dr and dc:
        return False
    return (not dr or start[1] % 2 == 1) and (not dc or start[0] % 2 == 1)


def parity_route_moves(start, end, config=None):
    """Route between AOD sites along odd rows/columns, as displacements.

    Compare paths through adjacent odd-odd intersections. A same-row shortcut
    is legal only on an odd row (similarly for an odd column).
    """
    if not (_is_valid_aod_position(start) and _is_valid_aod_position(end)):
        raise ValueError("Highway route endpoints must be AOD sites.")
    if start == end:
        return []

    def intersections(point):
        r, c = point
        if r % 2 and c % 2:
            return [point]
        if r % 2:
            return [(r, c - 1), (r, c + 1)]
        return [(r - 1, c), (r + 1, c)]

    routes = []
    if _highway_segment(start, end):
        routes.append([_movement_vector(start, end)])
    for first in intersections(start):
        for last in intersections(end):
            for corner in [(first[0], last[1]), (last[0], first[1])]:
                points = [start, first, corner, last, end]
                route = []
                for a, b in zip(points, points[1:]):
                    step = _movement_vector(a, b)
                    if step == (0, 0):
                        continue
                    if route and (
                        (route[-1][0] * step[0] > 0 and route[-1][1] == step[1] == 0)
                        or (route[-1][1] * step[1] > 0 and route[-1][0] == step[0] == 0)
                    ):
                        route[-1] = (route[-1][0] + step[0], route[-1][1] + step[1])
                    else:
                        route.append(step)
                routes.append(route)
    def key(route):
        cost = sum(_step_seconds(step, config) for step in route) if config else sum(
            abs(dr) + abs(dc) for dr, dc in route
        )
        return cost, len(route), tuple(route)
    return min(routes, key=key)


_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class _SideTour:
    load_step: tuple[int, int]
    visit_offsets: tuple[tuple[int, int], ...]
    routes: tuple[tuple[tuple[int, int], ...], ...]
    return_steps: tuple[tuple[int, int], ...]
    motion_seconds: float


def _optimize_visit_sides(vectors, config, *, start_offset=None):
    """Minimum parity-routed motion for a FIXED sequence of home-relative targets.

    Keep four endpoint states per visit; compare 4 x 4 transitions between
    adjacent visits, plus four loading/unloading directions at the boundaries.
    This is O(16*n) routing comparisons and O(n) storage, not a new subset TSP.
    All homes are even-even, so routing the shared offset has the same parity
    geometry as routing any loaded atom. Transfers are one cardinal step.
    A non-None start_offset means the load is already in the AOD: start there
    without another transfer, but still price the eventual return to home.
    """
    if not vectors:
        return _SideTour((0, 0), (), (), (), 0.0)
    if any(r % 2 or c % 2 for r, c in vectors):
        raise ValueError("Side optimization requires even-even home-relative targets.")
    if start_offset is not None and not _is_valid_aod_position(start_offset):
        raise ValueError("A reused load must start at an AOD offset.")
    points = [tuple((r + dr, c + dc) for dr, dc in _CARDINAL) for r, c in vectors]

    def edge(start, end):
        return _route(start, end, config)

    transfer_costs = [_step_seconds(step, config) for step in _CARDINAL]
    costs, starts = [], []
    for end in points[0]:
        if start_offset is None:
            cost, start = min((transfer_costs[s] + edge(offset, end)[0], s)
                              for s, offset in enumerate(_CARDINAL))
        else:
            cost, start = edge(start_offset, end)[0], None
        costs.append(cost)
        starts.append(start)
    parents = []
    for previous, current in zip(points, points[1:]):
        next_costs, predecessors = [], []
        for end in current:
            cost, predecessor = min((costs[s] + edge(start, end)[0], s)
                                    for s, start in enumerate(previous))
            next_costs.append(cost)
            predecessors.append(predecessor)
        parents.append(predecessors)
        costs = next_costs
    total, last, exit_side = min(
        (costs[s] + edge(start, end)[0] + transfer_costs[e], s, e)
        for s, start in enumerate(points[-1]) for e, end in enumerate(_CARDINAL)
    )
    chosen = [last]
    for predecessors in reversed(parents):
        chosen.append(predecessors[chosen[-1]])
    chosen.reverse()
    load_step = _CARDINAL[starts[chosen[0]]] if start_offset is None else (0, 0)
    initial = load_step if start_offset is None else start_offset
    offsets = tuple(options[side] for options, side in zip(points, chosen))
    routes = tuple(edge(start, end)[1] for start, end in zip((initial, *offsets[:-1]), offsets))
    unload_offset = _CARDINAL[exit_side]
    return_steps = edge(offsets[-1], unload_offset)[1] + (tuple(-v for v in unload_offset),)
    result = _SideTour(load_step, offsets, routes, return_steps, total)
    return result


def _translate(ids, positions, step, events, moves):
    """Apply and record one shared displacement; positions are mutable copies."""
    if step == (0, 0):
        return
    dr, dc = step
    for i, qid in enumerate(ids):
        start = positions[i]
        end = (start[0] + dr, start[1] + dc)
        events.append(("move", qid, start, end))
        positions[i] = end
    moves.append(step)


def _start_(vectors, moves, ids, positions, events):
    """Load a home group by a shared cardinal displacement, or retain its load."""
    if not ids:
        return vectors[:]
    if _is_valid_aod_position(positions[0]):
        if not all(_is_valid_aod_position(p) for p in positions):
            raise ValueError("An AOD load cannot mix loaded and trapped atoms.")
        return vectors[:]
    if any(_is_valid_aod_position(p) for p in positions):
        raise ValueError("A new load must start entirely in SLM traps.")
    first = vectors[0]
    step = min(_CARDINAL, key=lambda d: math.hypot(first[0] - d[0], first[1] - d[1]))
    _translate(ids, positions, step, events, moves)
    return [(dr - step[0], dc - step[1]) for dr, dc in vectors]


def _gate_parts(line):
    for statement in line.split(";"):
        statement = statement.strip()
        if statement:
            ids = tuple(map(int, re.findall(r"q\[(\d+)\]", statement)))
            pulse = statement.split(" q[", 1)[0]
            yield statement + ";", pulse, ids


def _append_gate(events, node):
    line = format_node_line(node)
    _, pulse, ids = next(_gate_parts(line))
    if events and events[-1][0] == "gate":
        previous = list(_gate_parts(events[-1][1]))
        if all(p == pulse and not set(qs) & set(ids) for _, p, qs in previous):
            events[-1] = ("gate", events[-1][1] + " " + line)
            return
    events.append(("gate", line))


def _shuttle_(vectors, moves, ids, positions, nodes, events, config):
    """Follow gate targets using actual positions and shared displacements.

    vectors are partner-minus-mover at entry, not differences of successive
    target vectors. Targets remain stationary throughout this legal AOD load.
    """
    targets = [
        (positions[i][0] + dr, positions[i][1] + dc)
        for i, (dr, dc) in enumerate(vectors)
    ]
    loaded = set(ids)
    for i, node in enumerate(nodes):
        gate_ids = op_node_signature(node)[2]
        if len(set(gate_ids) & loaded) != 1 or ids[i] not in gate_ids:
            raise ValueError("Each interaction must have exactly one loaded operand.")
        target = targets[i]
        if _is_valid_aod_position(target):
            raise ValueError("The interaction partner must remain in an SLM trap.")
        current = positions[i]
        if sum(abs(v) for v in _movement_vector(current, target)) != 1:
            options = []
            for dr, dc in _CARDINAL:
                end = (target[0] + dr, target[1] + dc)
                route = parity_route_moves(current, end, config)
                displacement = _movement_vector(current, end)
                ready = sum(
                    abs(targets[j][0] - positions[j][0] - displacement[0])
                    + abs(targets[j][1] - positions[j][1] - displacement[1]) == 1
                    for j in range(i + 1, len(nodes))
                )
                options.append((sum(_step_seconds(s, config) for s in route), -ready, route))
            route = min(options, key=lambda option: (option[0], option[1], tuple(option[2])))[2]
            for step in route:
                _translate(ids, positions, step, events, moves)
        if sum(abs(v) for v in _movement_vector(positions[i], target)) != 1:
            raise ValueError("Attempted to emit a gate outside interaction separation.")
        _append_gate(events, node)


def _return_(ids, moves, positions, homes, events, config):
    """Return the complete rigid AOD load to its own homes and unload once."""
    if not ids:
        return
    if not (len(ids) == len(positions) == len(homes)):
        raise ValueError("Return inputs must have matching lengths.")
    offsets = {_movement_vector(home, pos) for home, pos in zip(homes, positions)}
    if len(offsets) != 1:
        raise ValueError("Loaded atoms do not have a common displacement from home.")
    if positions == homes:
        return
    options = []
    for dr, dc in _CARDINAL:
        end = (homes[0][0] + dr, homes[0][1] + dc)
        route = parity_route_moves(positions[0], end, config)
        route = route + [(-dr, -dc)]
        options.append(route)
    route = min(options, key=lambda steps: (
        sum(_step_seconds(s, config) for s in steps), len(steps), tuple(steps)
    ))
    for step in route:
        _translate(ids, positions, step, events, moves)
    if positions != homes:
        raise ValueError("Return failed to restore home positions.")


def _event_batches(events):
    """Use exactly the movement compression/batching used by the text writer."""
    from .scheduling import _compress_linear_moves, _move_batch_key
    event_list = _compress_linear_moves(list(events))
    i = 0
    while i < len(event_list):
        first = event_list[i]
        i += 1
        batch = [first]
        if first[0] != "gate":
            key = _move_batch_key(first)
            used = {first[1]}
            while i < len(event_list):
                nxt = event_list[i]
                if nxt[0] == "gate" or _move_batch_key(nxt) != key or nxt[1] in used:
                    break
                batch.append(nxt)
                used.add(nxt[1])
                i += 1
        yield batch


def schedule_duration(events, config):
    """One common duration model for optimized, reset, and baseline schedules."""
    seconds = 0.0
    transfers = 0
    one_pulses = two_pulses = measurement_batches = 0
    for batch in _event_batches(events):
        first = batch[0]
        if first[0] == "gate":
            pulses = {(pulse, len(ids)) for _, pulse, ids in _gate_parts(first[1])}
            for pulse, arity in pulses:
                if pulse == "measure":
                    measurement_batches += 1
                elif arity == 1:
                    one_pulses += 1
                elif arity == 2:
                    two_pulses += 1
                else:
                    raise ValueError("Only one- and two-qubit pulses are supported.")
        else:
            seconds += max(_step_seconds(_movement_vector(e[2], e[3]), config) for e in batch)
            # A step whose atoms are all held changes no traps.
            transfers += int(any(
                e[0] != "hold" and not (_is_valid_aod_position(e[2]) and _is_valid_aod_position(e[3]))
                for e in batch
            ))
    return (
        seconds * config["t_switch"].to("seconds").units
        + transfers * config["transfer_SLM_AOD"]
        + one_pulses * (config["average_single_gate_time"] + config["t_switch"])
        + two_pulses * (config["average_two_gate_time"] + config["t_switch"])
        + measurement_batches * (
            config.get("measurement_time", 0 * config["t_switch"]) + config["t_switch"]
        )
    )


@dataclass
class _Plan:
    positions: dict[int, tuple[int, int]]
    ids: list[int]
    events: list[ScheduleEvent]
    reused_groups: int = 0

    def copy(self):
        return _Plan(dict(self.positions), list(self.ids), list(self.events), self.reused_groups)


def _return_plan(plan, homes, config):
    if plan.ids:
        positions = [plan.positions[q] for q in plan.ids]
        _return_(plan.ids, [], positions, [homes[q] for q in plan.ids], plan.events, config)
        plan.positions.update(zip(plan.ids, positions))
        plan.ids = []


def _reuse_group_allowed(previous_ids, current_ids, horizon, stage_index, next_use_by_id):
    previous, current = set(previous_ids), set(current_ids)
    if not previous or not current or not current <= previous:
        return False
    # Compatibility: zero still permits identical-set continuation.
    if horizon == 0:
        return current == previous
    if horizon == float("inf") or horizon == "Inf":
        return True
    return all(
        q in next_use_by_id and next_use_by_id[q] - stage_index <= horizon
        for q in previous - current
    )


def _preferred_mover(pair, config):
    """Return the ancilla operand when exactly one operand is an ancilla."""
    if not config.get("prefer_moving_ancillas", False):
        return None
    ancillas = config.get("_ancilla_qubits", frozenset())
    preferred = [q for q in pair if q in ancillas]
    return preferred[0] if len(preferred) == 1 else None


def _reuse_movers(nodes, previous_ids, config):
    """Single legality predicate used for both reuse selection and execution."""
    previous = set(previous_ids)
    movers = []
    for node in nodes:
        operands = set(op_node_signature(node)[2])
        loaded = operands & previous
        if len(operands) != 2 or len(loaded) != 1:
            return None
        mover = next(iter(loaded))
        preferred = _preferred_mover(operands, config)
        if preferred is not None and mover != preferred:
            return None
        movers.append(mover)
    return movers


def _build_groups(nodes, positions, config):
    """Group only in the state that will actually be used after a reset."""
    groups = []
    spans = _max_grid_spans(config["max_dimension"])
    for node in nodes:
        pair = op_node_signature(node)[2]
        if len(pair) != 2:
            raise ValueError("A two-qubit layer must contain only two-qubit gates.")
        choices = [(q, _movement_vector(positions[q], positions[pair[1 - i]])) for i, q in enumerate(pair)]
        preferred = _preferred_mover(pair, config)
        if preferred is not None:
            choices = [choice for choice in choices if choice[0] == preferred]
        placed = False
        if config.get("parallel", False):
            for group, movers, vectors in groups:
                fits = [
                    (q, vector) for q, vector in choices
                    if _fits_same_aod([positions[m] for m in movers], positions[q], *spans)
                    and _vector_alignment_score(vectors, vector) >= config.get("alignment_conc", 0)
                ]
                if fits:
                    q, vector = max(fits, key=lambda item: _vector_alignment_score(vectors, item[1]))
                    group.append(node)
                    movers.append(q)
                    vectors.append(vector)
                    placed = True
                    break
        if not placed:
            q, vector = choices[0]
            groups.append(([node], [q], [vector]))

    #now we order the movers based on displacement
    for group, movers, vectors in groups:
        ordered = sorted(
            zip(group, movers, vectors),
            key=lambda item: (item[2][0], item[2][1], item[1]),
        )
        group[:] = [item[0] for item in ordered]
        movers[:] = [item[1] for item in ordered]
        vectors[:] = [item[2] for item in ordered]

    return [(group, movers) for group, movers, _ in groups]


def _execute_group(plan, nodes, movers, homes, config, reuse=False):
    if reuse:
        if _reuse_movers(nodes, plan.ids, config) != movers:
            raise ValueError("Cannot reuse an AOD load containing both interacting operands.")
    elif plan.ids:
        raise ValueError("A new group requires the preceding load to be returned.")
    offsets = {
        _movement_vector(homes[q], plan.positions[q])
        for q in set(movers) | set(plan.ids)
    }
    if len(offsets) != 1:
        raise ValueError("Group movers must share one rigid displacement from home.")
    start_offset = offsets.pop()
    vectors = []
    home_targets = []
    for node, mover in zip(nodes, movers):
        pair = op_node_signature(node)[2]
        partner = pair[1] if pair[0] == mover else pair[0]
        vectors.append(_movement_vector(plan.positions[mover], plan.positions[partner]))
        home_targets.append(_movement_vector(homes[mover], plan.positions[partner]))
    # Here we use Held-Karp when there are 24 or fewer unique displacements.
    # Optimize home-relative targets, but shuttle with current-relative vectors.
    order = _order_group_displacements(home_targets, config, start_offset=start_offset)

    nodes = [nodes[i] for i in order]
    ids = [movers[i] for i in order]
    vectors = [vectors[i] for i in order]
    ids.extend(q for q in plan.ids if q not in ids)
    positions = [plan.positions[q] for q in ids]
    moves = []
    if config.get("_side_ordering") == "greedy":
        vectors = _start_(vectors, moves, ids, positions, plan.events)
        _shuttle_(vectors, moves, ids, positions, nodes, plan.events, config)
    else:
        # Independent gates may be routed in any order, regardless of gate type
        # or CZ preprocessing. Shared-wire order remains fixed by the layers.
        tour = _optimize_visit_sides(
            [home_targets[i] for i in order], config,
            start_offset=start_offset if reuse else None,
        )
        _translate(ids, positions, tour.load_step, plan.events, moves)
        loaded = set(ids)
        for i, (node, steps) in enumerate(zip(nodes, tour.routes)):
            for step in steps:
                _translate(ids, positions, step, plan.events, moves)
            pair = op_node_signature(node)[2]
            if set(pair) & loaded != {ids[i]}:
                raise ValueError("Each interaction must have exactly one loaded operand.")
            partner = pair[1] if pair[0] == ids[i] else pair[0]
            target = plan.positions[partner]
            if _is_valid_aod_position(target) or sum(abs(v) for v in _movement_vector(positions[i], target)) != 1:
                raise ValueError("Side-planned interaction must be adjacent to an SLM partner.")
            _append_gate(plan.events, node)
        # The return is a cost-to-go estimate. Keep the load active so the next
        # layer can reuse it; _return_plan executes the return when needed.
    plan.positions.update(zip(ids, positions))
    plan.ids = ids
    plan.reused_groups += int(reuse)


def _reset_layer(plan, nodes, homes, config):
    if not nodes:
        return
    _return_plan(plan, homes, config)
    # All grouping decisions now see the state after the proposed return.
    for group, movers in _build_groups(nodes, plan.positions, config):
        _return_plan(plan, homes, config)
        _execute_group(plan, group, movers, homes, config)


def _closed_cost(plan, homes, config):
    ending = _Plan(dict(plan.positions), list(plan.ids), [])
    _return_plan(ending, homes, config)
    return schedule_duration(plan.events + ending.events, config)


def _plan_layer(nodes, initial, homes, config, stage, next_use, allow_reuse=True):
    reset = initial.copy()
    _reset_layer(reset, nodes, homes, config)
    if not allow_reuse or not initial.ids or not nodes:
        return reset
    # Gates in a DAG layer are disjoint. Only gates with one loaded operand
    # can precede a reset; AOD-AOD gates remain in the reset portion.
    eligible = [n for n in nodes if _reuse_movers([n], initial.ids, config) is not None]
    if not config.get("parallel", False):
        eligible = eligible[:1]
    movers = _reuse_movers(eligible, initial.ids, config)
    if not movers or not _reuse_group_allowed(
        initial.ids, movers, config.get("T_reuse", 0), stage, next_use
    ):
        return reset
    reuse = initial.copy()
    _execute_group(reuse, eligible, movers, homes, config, reuse=True)
    selected = {id(n) for n in eligible}
    _reset_layer(reuse, [n for n in nodes if id(n) not in selected], homes, config)
    # Compare equal endpoints (all atoms at home). The closure is a cost-to-go
    # estimate, not prematurely committed: the next layer can still reuse.
    return reuse if _closed_cost(reuse, homes, config) < _closed_cost(reset, homes, config) else reset


def best_path_for_layer(
    layer_nodes: list[DAGOpNode], qubits: list[Qubit], config: dict,
    event_log: list[ScheduleEvent], Previous_Ids=None,
    Previous_Positions=None, current_positions=None, stage_index=0,
    next_use_by_id=None,
):
    """Compatibility layer API; full-circuit guarantees use schedule_circuit."""
    config = dict(config, _motion_parameters=_motion_parameters(config))
    homes = {q.id: q.grid_position() for q in qubits}
    positions = dict(homes if current_positions is None else current_positions)
    ids = list(Previous_Ids or [])
    if ids:
        if Previous_Positions is None or len(ids) != len(Previous_Positions):
            raise ValueError("Previous IDs and positions must have matching lengths.")
        positions.update(zip(ids, Previous_Positions))
    _check_layer(layer_nodes, homes)
    plan = _plan_layer(
        layer_nodes, _Plan(positions, ids, []), homes, config,
        stage_index, next_use_by_id or {},
    )
    event_log.extend(plan.events)
    if current_positions is not None:
        current_positions.update(plan.positions)
    from .scheduling import count_emitted_timesteps
    return (
        count_emitted_timesteps(plan.events), schedule_duration(plan.events, config),
        [plan.positions[q] for q in plan.ids], plan.ids,
    )


def _check_layer(nodes, homes):
    used = set()
    for node in nodes:
        ids = op_node_signature(node)[2]
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("Expected two distinct gate operands.")
        if not set(ids) <= homes.keys():
            raise ValueError("Gate operand is missing from the initial placement.")
        if used & set(ids):
            raise ValueError("A DAG layer must have disjoint gate operands.")
        used.update(ids)


def validate_schedule(events, homes, config, reference_nodes=None, require_home=True):
    """Replay continuity, highway interiors, full loads, gates, and wire order.

    This is validation of the project's discrete rigid-AOD model, not a model
    of finite trap extent, laser crosstalk, or physical AOD travel bounds.
    """
    if len(set(homes.values())) != len(homes) or any(_is_valid_aod_position(p) for p in homes.values()):
        raise ValueError("Homes must be unique even-even SLM sites.")
    positions = dict(homes)
    active = set()
    traces = defaultdict(list)
    for batch in _event_batches(events):
        if batch[0][0] == "gate":
            used = set()
            for statement, _, ids in _gate_parts(batch[0][1]):
                if not ids or not set(ids) <= positions.keys() or used & set(ids):
                    raise ValueError("Invalid operands or overlapping simultaneous gates.")
                used.update(ids)
                if len(ids) == 2:
                    if len(set(ids) & active) != 1:
                        raise ValueError("A two-qubit gate requires one AOD and one SLM atom.")
                    preferred = _preferred_mover(ids, config)
                    if preferred is not None and preferred not in active:
                        raise ValueError("A data-ancilla gate must move its ancilla operand.")
                    if sum(abs(v) for v in _movement_vector(positions[ids[0]], positions[ids[1]])) != 1:
                        raise ValueError("Two-qubit gate separation is not one grid step.")
                elif len(ids) != 1:
                    raise ValueError("Only one- and two-qubit gates are supported.")
                for q in ids:
                    traces[q].append(statement)
            continue
        moved = {e[1] for e in batch}
        # Held atoms stay in the AOD while a transfer step carries them over
        # their SLM home; only the other atoms of the step change traps.
        held = {e[1] for e in batch if e[0] == "hold"}
        if any(e[0] not in {"move", "hold"} for e in batch):
            raise ValueError("Unknown movement event kind.")
        first = batch[0]
        loading = not _is_valid_aod_position(first[2])
        unloading = not _is_valid_aod_position(first[3])
        if loading and unloading:
            raise ValueError("Direct SLM-to-SLM movement is forbidden.")
        if held and not (loading or unloading):
            raise ValueError("Held atoms may only ride a load or unload step.")
        if not held <= active:
            raise ValueError("Only atoms already in the AOD can be held.")
        if loading:
            if active != held:
                raise ValueError("Load attempted before returning the existing AOD load.")
            span_positions = [e[2] for e in batch]
            if not all(_fits_same_aod(span_positions[:i], p, *_max_grid_spans(config["max_dimension"]))
                       for i, p in enumerate(span_positions)):
                raise ValueError("AOD load exceeds the configured span.")
            active = moved.copy()
        elif moved != active:
            raise ValueError("The entire AOD load must move or unload together.")
        for kind, q, start, end in batch:
            if q not in positions or positions[q] != start:
                raise ValueError("Movement continuity violation.")
            if loading or unloading:
                if sum(abs(v) for v in _movement_vector(start, end)) != 1:
                    raise ValueError("Transfers must be one cardinal grid step.")
                if loading and _is_valid_aod_position(start):
                    raise ValueError("Load must start in an SLM trap.")
                if unloading and _is_valid_aod_position(end):
                    raise ValueError("Unload must end in an SLM trap.")
            elif not _highway_segment(start, end):
                raise ValueError("Movement crosses an SLM site or leaves the highway.")
            positions[q] = end
        if len({_movement_vector(e[2], e[3]) for e in batch}) != 1:
            raise ValueError("The AOD batch must share one displacement.")
        if len(set(positions.values())) != len(positions):
            raise ValueError("Atom position collision.")
        if unloading:
            active = held
    if require_home and (active or positions != homes):
        raise ValueError("The completed circuit must return every atom to its home.")
    if reference_nodes is not None:
        _validate_barrier_order(events, reference_nodes)
        expected = defaultdict(list)
        commuting, fixed = set(), set()
        for node in reference_nodes:
            if node.op.name in {"swap", "barrier"}:
                continue
            if len(node.qargs) not in (1, 2):
                continue
            line = format_node_line(node)
            (commuting if _is_reorderable_cz(node) else fixed).add(line)
            for q in op_node_signature(node)[2]:
                expected[q].append(line)
        if not _cz_commutation_enabled(config):
            commuting = set()
        commuting -= fixed
        if _cz_run_traces(traces, commuting) != _cz_run_traces(expected, commuting):
            raise ValueError("Gate identities, multiplicities, or per-qubit order changed.")
    return positions


def _cz_commutation_enabled(config):
    """CZ preprocessing opts in to reordering commuting unconditional CZs."""
    return bool(config.get("reorder_cz", False) or config.get("reorder_cz_by_displacement", False))


def _cz_run_traces(traces, commuting):
    """Per-wire traces where each maximal run of ``commuting`` CZs is a multiset.

    Unconditional CZs are diagonal and commute; any other operation on the
    wire ends a run, as in ``dag_helper._validate_cz_reordering``.
    """
    result = {}
    for q, trace in traces.items():
        canonical, run = [], []
        for statement in trace:
            if statement in commuting:
                run.append(statement)
            else:
                canonical.extend((tuple(sorted(run)), statement))
                run = []
        canonical.append(tuple(sorted(run)))
        result[q] = canonical
    return result


def _validate_barrier_order(events, reference_nodes):
    """Require every emitted operation to remain in its QASM barrier epoch."""
    epochs = [defaultdict(int)]
    for node in reference_nodes:
        if node.op.name == "swap":
            continue
        if node.op.name == "barrier":
            epochs.append(defaultdict(int))
        elif len(node.qargs) in (1, 2):
            epochs[-1][format_node_line(node)] += 1

    epoch = 0
    for event in events:
        if event[0] != "gate":
            continue
        for statement, _, _ in _gate_parts(event[1]):
            while epoch < len(epochs) - 1 and not any(epochs[epoch].values()):
                epoch += 1
            if epochs[epoch].get(statement, 0):
                epochs[epoch][statement] -= 1
            elif any(later.get(statement, 0) for later in epochs[epoch + 1:]):
                raise ValueError("An operation crossed a QASM barrier.")


def _optimize_retained_loads(events, layers, homes, config, singles=None):
    """Jointly route actual load-to-unload intervals without changing retention.

    The ordinary planner has already selected legal loads under T_reuse. Here
    only their internal routes change. Layer boundaries and intervening 1Q
    contexts remain ordered, and no atom is added to or removed from a load.
    With CZ preprocessing enabled, consecutive CZ-only layers with nothing
    between them (``singles`` empty, no barrier-emptied layer) commute and form
    one routing stage, so their gates may be visited in any order.
    """
    merge = singles is not None and _cz_commutation_enabled(config)
    cz_layers = [bool(nodes) and all(map(_is_reorderable_cz, nodes)) for nodes in layers]

    def same_stage(previous, layer):
        return merge and previous is not None and previous < layer and cz_layers[previous] and all(
            cz_layers[k] and not singles[k] for k in range(previous + 1, layer + 1)
        )

    occurrences = defaultdict(deque)
    for stage, nodes in enumerate(layers):
        for node in nodes:
            occurrences[format_node_line(node)].append((stage, node))

    def reroute(interval, ids):
        visits, targets, labels, contexts = [], [], [], {}
        pending = []
        stage, last_layer = -1, None
        original_layers = set()
        loaded = set(ids)
        for event in interval:
            if event[0] != "gate":
                continue
            parts = list(_gate_parts(event[1]))
            if all(len(pair) == 1 for _, _, pair in parts):
                pending.append(event)
                continue
            for statement, _, pair in parts:
                if len(pair) != 2 or not occurrences[statement]:
                    raise ValueError("Retained-load planning lost a gate occurrence.")
                layer, node = occurrences[statement].popleft()
                if pending or (layer != last_layer and not same_stage(last_layer, layer)):
                    stage += 1
                    contexts[stage] = pending
                    pending = []
                last_layer = layer
                movers = set(pair) & loaded
                if len(movers) != 1:
                    raise ValueError("Retained interaction must have exactly one loaded operand.")
                mover = next(iter(movers))
                partner = pair[1] if pair[0] == mover else pair[0]
                visits.append((node, mover, partner))
                targets.append(_movement_vector(homes[mover], homes[partner]))
                labels.append(stage)
                original_layers.add(layer)
        if len(original_layers) < 2:
            return interval

        order = _order_group_displacements(targets, config, layer_ids=labels)
        if sorted(order) != list(range(len(visits))) or [labels[i] for i in order] != sorted(labels):
            raise ValueError("Retained-load tour changed gate occurrences or layer order.")
        plan = _Plan(dict(homes), list(ids), [])

        def translate(step):
            positions = [plan.positions[q] for q in ids]
            _translate(ids, positions, step, plan.events, [])
            plan.positions.update(zip(ids, positions))

        greedy = config.get("_side_ordering") == "greedy"
        if greedy:
            positions = [homes[q] for q in ids]
            _start_([targets[order[0]]] * len(ids), [], ids, positions, plan.events)
            plan.positions.update(zip(ids, positions))
        else:
            tour = _optimize_visit_sides([targets[i] for i in order], config)
            translate(tour.load_step)
        previous_stage = None
        for position, index in enumerate(order):
            if labels[index] != previous_stage:
                plan.events.extend(contexts[labels[index]])
                previous_stage = labels[index]
            node, mover, partner = visits[index]
            if greedy:
                _execute_group(plan, [node], [mover], homes, config, reuse=True)
            else:
                for step in tour.routes[position]:
                    translate(step)
                if sum(abs(v) for v in _movement_vector(plan.positions[mover], homes[partner])) != 1:
                    raise ValueError("Retained-load visit is not adjacent to its partner.")
                _append_gate(plan.events, node)
        plan.events.extend(pending)
        if greedy:
            _return_plan(plan, homes, config)
        else:
            for step in tour.return_steps:
                translate(step)
            if plan.positions != homes:
                raise ValueError("Retained-load tour did not restore original homes.")
        return plan.events

    result, interval, ids = [], [], []
    for batch in _event_batches(events):
        first = batch[0]
        if first[0] == "move" and not _is_valid_aod_position(first[2]):
            if ids:
                raise ValueError("Cannot start a new load before unloading the previous load.")
            ids = [event[1] for event in batch]
            if any(start != homes[q] for _, q, start, _ in batch):
                raise ValueError("A retained-load tour must start at the original homes.")
        if ids:
            interval.extend(batch)
            if first[0] == "move" and not _is_valid_aod_position(first[3]):
                result.extend(reroute(interval, ids))
                interval, ids = [], []
        else:
            result.extend(batch)
    if ids or any(occurrences.values()):
        raise ValueError("Retained-load planning did not consume the complete circuit.")
    return result


def _hold_reloaded_atoms(events):
    """Keep atoms in the AOD through an unload that is followed by their reload.

    The planner releases a whole load before picking up the next one. An atom
    unloaded at home and loaded again by the very next transfer need not
    leave the AOD: it rides both rigid transfer steps as a "hold", passing
    over its home site, and only atoms that actually change traps are loaded
    or unloaded. Atoms measured or reset in between are released as before.
    """
    batches = [list(batch) for batch in _event_batches(events)]

    def transfer(batch, index):
        return batch[0][0] != "gate" and not _is_valid_aod_position(batch[0][index])

    for i, unload in enumerate(batches):
        if not transfer(unload, 3):
            continue
        j = next((k for k in range(i + 1, len(batches)) if batches[k][0][0] != "gate"), None)
        if j is None or not transfer(batches[j], 2):
            continue
        released = {
            q for batch in batches[i + 1:j]
            for _, pulse, ids in _gate_parts(batch[0][1])
            if pulse.startswith(("measure", "reset")) for q in ids
        }
        kept = ({e[1] for e in unload} & {e[1] for e in batches[j]}) - released
        for batch in (unload, batches[j]):
            batch[:] = [("hold", *e[1:]) if e[1] in kept else e for e in batch]
    return [event for batch in batches for event in batch]


def _layered_candidate(layers, singles, homes, config, allow_reuse):
    from .scheduling import single_qubit_layer_time
    next_uses = [{} for _ in layers]
    future = {}
    for i in reversed(range(len(layers))):
        next_uses[i] = dict(future)
        for node in layers[i]:
            future.update({q: i for q in op_node_signature(node)[2]})
    plan = _Plan(dict(homes), [], [])
    for i, nodes in enumerate(layers):
        lines, _ = single_qubit_layer_time(
            singles[i], config["average_single_gate_time"], config["t_switch"],
            config.get("measurement_time"),
        )
        plan.events.extend(("gate", line) for line in lines)
        layer_plan = _plan_layer(
            nodes, _Plan(dict(plan.positions), list(plan.ids), []),
            homes, config, i, next_uses[i], allow_reuse,
        )
        plan.events.extend(layer_plan.events)
        plan.positions, plan.ids = layer_plan.positions, layer_plan.ids
        plan.reused_groups += layer_plan.reused_groups
    lines, _ = single_qubit_layer_time(
        singles[-1], config["average_single_gate_time"], config["t_switch"],
        config.get("measurement_time"),
    )
    plan.events.extend(("gate", line) for line in lines)
    _return_plan(plan, homes, config)
    plan.events = _optimize_retained_loads(plan.events, layers, homes, config, singles)
    plan.events = _hold_reloaded_atoms(plan.events)
    return plan


def _sequential_candidate(ops, homes, config):
    """Home-return baseline with naive_dag's next-2Q-gate mover lookahead."""
    from .scheduling import single_qubit_layer_time
    plan = _Plan(dict(homes), [], [])
    two_indices = [i for i, n in enumerate(ops) if _is_two_qubit_operation(n)]
    next_pairs = {}
    for a, b in zip(two_indices, two_indices[1:]):
        next_pairs[a] = set(op_node_signature(ops[b])[2])
    i = 0
    while i < len(ops):
        node = ops[i]
        if not _is_two_qubit_operation(node):
            block = []
            while i < len(ops) and not _is_two_qubit_operation(ops[i]):
                block.append(ops[i])
                i += 1
            lines, _ = single_qubit_layer_time(
                block, config["average_single_gate_time"], config["t_switch"],
                config.get("measurement_time"),
            )
            plan.events.extend(("gate", line) for line in lines)
            continue
        pair = op_node_signature(node)[2]
        preferred = _preferred_mover(pair, config)
        if plan.ids and (plan.ids[0] not in pair or
                         (preferred is not None and plan.ids[0] != preferred)):
            _return_plan(plan, homes, config)
        following = next_pairs.get(i, set())
        mover = plan.ids[0] if plan.ids else (
            preferred if preferred is not None else
            (pair[1] if pair[1] in following else pair[0])
        )
        _execute_group(plan, [node], [mover], homes, config, reuse=bool(plan.ids))
        following_preferred = _preferred_mover(following, config)
        if mover not in following or (following_preferred is not None and following_preferred != mover):
            _return_plan(plan, homes, config)
        i += 1
    _return_plan(plan, homes, config)
    return plan


def _legacy_baseline(ops, homes, config):
    """Capture actual naive_dag event choices on independent placement objects.

    Ignore its legacy timer, restore initial homes, then retime/validate through
    the same model as every other candidate. Invalid legacy paths are rejected.
    """
    #TODO: fix naive_dag to avoid rule conflicts.

    from naive_dag import dynamics as legacy
    from naive_dag.grid import generate_grid, place_qubit
    from .scheduling import single_qubit_layer_time
    grid = generate_grid(config["dimensions"], config["rydberg_radius"])
    qubits = [place_qubit(grid, *homes[q], q) for q in sorted(homes)]
    events = []
    reused_groups = 0
    i = 0
    while i < len(ops):
        if _is_two_qubit_operation(ops[i]):
            pair = op_node_signature(ops[i])[2]
            reused_groups += int(any(
                q.id in pair and _is_valid_aod_position(q.grid_position()) for q in qubits
            ))
            _, _, emitted = legacy.best_path_for_gate(ops, i, qubits, grid, config, 0)
            # That API also emits trailing single-qubit gates using a different
            # packer. Keep only its movement and two-qubit pulse here; emit 1Q
            # operations once, below, with the common order-preserving packer.
            for event in emitted:
                if event[0] == "move" or any(len(ids) == 2 for _, _, ids in _gate_parts(event[1])):
                    events.append(event)
            i += 1
        else:
            block = []
            while i < len(ops) and not _is_two_qubit_operation(ops[i]):
                block.append(ops[i])
                i += 1
            lines, _ = single_qubit_layer_time(
                block, config["average_single_gate_time"], config["t_switch"],
                config.get("measurement_time"),
            )
            events.extend(("gate", line) for line in lines)
    positions = dict(homes)
    for event in events:
        if event[0] == "move":
            positions[event[1]] = event[3]
    # Usually there is one vacant original home and at most one displaced atom.
    # More general relocations are accepted only when homes can be restored
    # without occupying another atom's site; otherwise this candidate is invalid.
    while positions != homes:
        movable = [q for q in positions if positions[q] != homes[q] and homes[q] not in positions.values()]
        if not movable:
            raise ValueError("Legacy placement cannot be restored without an occupied home.")
        loaded = [q for q, p in positions.items() if _is_valid_aod_position(p)]
        q = loaded[0] if loaded else movable[0]
        if q not in movable:
            raise ValueError("Legacy loaded atom cannot return to its own home.")
        current = [positions[q]]
        if not _is_valid_aod_position(current[0]):
            _start_([_movement_vector(current[0], homes[q])], [], [q], current, events)
        _return_([q], [], current, [homes[q]], events, config)
        positions[q] = current[0]
    return _Plan(positions, [], events, reused_groups)


@dataclass
class CircuitSchedule:
    events: list[ScheduleEvent]
    duration: Any
    selected: str
    candidate_times: dict[str, Any]
    rejected_candidates: dict[str, str]
    reused_groups: int


def schedule_circuit(layers, singles, qubits, config, reference_nodes=None):
    """Build and validate one complete schedule using the configured policy.

    Held-Karp and side DP apply to ordinary groups regardless of CZ reordering.
    Only an explicit greedy side policy disables side DP. T_reuse controls load
    retention; failures propagate instead of silently switching scheduling policy.
    """
    config = dict(config, _motion_parameters=_motion_parameters(config))
    layers = [list(nodes) for nodes in layers]
    if len(singles) != len(layers) + 1:
        raise ValueError("Expected one single-qubit context bucket per layer boundary.")
    homes = {q.id: q.grid_position() for q in qubits}
    for nodes in layers:
        _check_layer(nodes, homes)
    if reference_nodes is None:
        reference_nodes = [
            node for i, layer in enumerate(layers) for node in list(singles[i]) + layer
        ] + list(singles[-1])
    reference_nodes = list(reference_nodes)
    config["_ancilla_qubits"] = frozenset(
        q for node in reference_nodes if node.op.name == "reset"
        for q in op_node_signature(node)[2]
    )
    selected = "greedy_reuse" if config.get("_side_ordering") == "greedy" else "reuse"
    plan = _layered_candidate(layers, singles, homes, config, True)
    validate_schedule(plan.events, homes, config, reference_nodes)
    duration = schedule_duration(plan.events, config)
    return CircuitSchedule(
        plan.events, duration, selected, {selected: duration}, {}, plan.reused_groups,
    )
