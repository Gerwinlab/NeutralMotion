from __future__ import annotations

import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import format_node_line, op_node_signature
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


def _held_karp_displacement_order(
    vectors: list[tuple[int, int]], config: dict,
    *, start_offset: tuple[int, int] = (0, 0),
) -> list[int]:
    """Return all vector indices in an exact minimum-cost displacement tour.

    Cities are partner positions relative to each mover's immutable home.
    The path starts at the current rigid-load offset and ends at home (0, 0),
    not at the starting offset of a reused load. Edge weights use the configured
    movement-time model, approximating targets without interaction-side routing.
    Repeated displacements are expanded consecutively while retaining their
    original relative index order.
    """
    if not vectors:
        return []

    indices_by_displacement: dict[tuple[int, int], list[int]] = {}
    for index, vector in enumerate(vectors):
        indices_by_displacement.setdefault(vector, []).append(index)
    unique = list(indices_by_displacement)
    count = len(unique)

    def edge_cost(start: tuple[int, int], end: tuple[int, int]) -> float:
        return _step_seconds((end[0] - start[0], end[1] - start[1]), config)

    depot = (0, 0)
    start_costs = [edge_cost(start_offset, vector) for vector in unique]
    return_costs = [edge_cost(vector, depot) for vector in unique]
    costs = [
        [edge_cost(start, end) for end in unique]
        for start in unique
    ]

    @lru_cache(maxsize=None)
    def solve(mask: int, last: int) -> tuple[float, int]:
        """Minimum start-to-``last`` path visiting exactly ``mask``."""
        last_bit = 1 << last
        if mask == last_bit:
            return start_costs[last], -1
        previous_mask = mask ^ last_bit
        return min(
            (
                solve(previous_mask, previous)[0] + costs[previous][last],
                previous,
            )
            for previous in range(count)
            if previous_mask & (1 << previous)
        )

    full_mask = (1 << count) - 1
    _, last = min(
        (
            solve(full_mask, candidate)[0] + return_costs[candidate],
            candidate,
        )
        for candidate in range(count)
    )

    reverse_unique_order = []
    mask = full_mask
    while last != -1:
        reverse_unique_order.append(last)
        _, previous = solve(mask, last)
        mask ^= 1 << last
        last = previous
    unique_order = reversed(reverse_unique_order)
    solve.cache_clear()

    return [
        vector_index
        for unique_index in unique_order
        for vector_index in indices_by_displacement[unique[unique_index]]
    ]


def _order_group_displacements(
    vectors: list[tuple[int, int]], config: dict,
    *, start_offset: tuple[int, int] = (0, 0),
) -> list[int]:
    """Order home-relative targets, preserving current-relative alignment."""
    unique_count = len(set(vectors))
    if unique_count <= _MAX_EXACT_DISPLACEMENT_TOUR:
        return _held_karp_displacement_order(vectors, config, start_offset=start_offset)

    warnings.warn(
        f"Group has {unique_count} unique displacements; exact Held-Karp "
        f"ordering is limited to {_MAX_EXACT_DISPLACEMENT_TOUR}. Falling "
        "back to alignment ordering.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _sort_group_by_alignment([
        (vector[0] - start_offset[0], vector[1] - start_offset[1])
        for vector in vectors
    ])


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

    @lru_cache(maxsize=None)
    def edge(start, end):
        route = tuple(parity_route_moves(start, end, config))
        return sum(_step_seconds(step, config) for step in route), route

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
    edge.cache_clear()
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
        if first[0] == "move":
            key = _move_batch_key(first)
            used = {first[1]}
            while i < len(event_list):
                nxt = event_list[i]
                if nxt[0] != "move" or _move_batch_key(nxt) != key or nxt[1] in used:
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
            transfers += int(
                not _is_valid_aod_position(first[2]) or not _is_valid_aod_position(first[3])
            )
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
        first = batch[0]
        loading = not _is_valid_aod_position(first[2])
        unloading = not _is_valid_aod_position(first[3])
        if loading and unloading:
            raise ValueError("Direct SLM-to-SLM movement is forbidden.")
        if loading:
            if active:
                raise ValueError("Load attempted before returning the existing AOD load.")
            span_positions = [e[2] for e in batch]
            if not all(_fits_same_aod(span_positions[:i], p, *_max_grid_spans(config["max_dimension"]))
                       for i, p in enumerate(span_positions)):
                raise ValueError("AOD load exceeds the configured span.")
            active = moved.copy()
        elif moved != active:
            raise ValueError("The entire AOD load must move or unload together.")
        for _, q, start, end in batch:
            if q not in positions or positions[q] != start:
                raise ValueError("Movement continuity violation.")
            if loading or unloading:
                if sum(abs(v) for v in _movement_vector(start, end)) != 1:
                    raise ValueError("Transfers must be one cardinal grid step.")
                if loading and (q in positions and _is_valid_aod_position(start)):
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
            active = set()
    if require_home and (active or positions != homes):
        raise ValueError("The completed circuit must return every atom to its home.")
    if reference_nodes is not None:
        _validate_barrier_order(events, reference_nodes)
        expected = defaultdict(list)
        for node in reference_nodes:
            if node.op.name in {"swap", "barrier"}:
                continue
            if len(node.qargs) not in (1, 2):
                continue
            line = format_node_line(node)
            for q in op_node_signature(node)[2]:
                expected[q].append(line)
        if dict(traces) != dict(expected):
            raise ValueError("Gate identities, multiplicities, or per-qubit order changed.")
    return positions


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
