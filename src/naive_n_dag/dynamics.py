from __future__ import annotations

import math
import re
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
    return _profile_seconds(math.hypot(*step) * spacing, velocity, acceleration)


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
    one_pulses = two_pulses = 0
    for batch in _event_batches(events):
        first = batch[0]
        if first[0] == "gate":
            pulses = {(pulse, len(ids)) for _, pulse, ids in _gate_parts(first[1])}
            for _, arity in pulses:
                if arity == 1:
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


def _reuse_movers(nodes, previous_ids):
    """Single legality predicate used for both reuse selection and execution."""
    previous = set(previous_ids)
    movers = []
    for node in nodes:
        operands = set(op_node_signature(node)[2])
        loaded = operands & previous
        if len(operands) != 2 or len(loaded) != 1:
            return None
        movers.append(next(iter(loaded)))
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
    return [(group, movers) for group, movers, _ in groups]


def _execute_group(plan, nodes, movers, config, reuse=False):
    if reuse:
        if _reuse_movers(nodes, plan.ids) != movers:
            raise ValueError("Cannot reuse an AOD load containing both interacting operands.")
    elif plan.ids:
        raise ValueError("A new group requires the preceding load to be returned.")
    vectors = []
    for node, mover in zip(nodes, movers):
        pair = op_node_signature(node)[2]
        partner = pair[1] if pair[0] == mover else pair[0]
        vectors.append(_movement_vector(plan.positions[mover], plan.positions[partner]))
    order = _sort_group_by_alignment(vectors)
    nodes = [nodes[i] for i in order]
    ids = [movers[i] for i in order]
    vectors = [vectors[i] for i in order]
    ids.extend(q for q in plan.ids if q not in ids)
    positions = [plan.positions[q] for q in ids]
    moves = []
    vectors = _start_(vectors, moves, ids, positions, plan.events)
    _shuttle_(vectors, moves, ids, positions, nodes, plan.events, config)
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
        _execute_group(plan, group, movers, config)


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
    eligible = [n for n in nodes if _reuse_movers([n], initial.ids) is not None]
    if not config.get("parallel", False):
        eligible = eligible[:1]
    movers = _reuse_movers(eligible, initial.ids)
    if not movers or not _reuse_group_allowed(
        initial.ids, movers, config.get("T_reuse", 0), stage, next_use
    ):
        return reset
    reuse = initial.copy()
    _execute_group(reuse, eligible, movers, config, reuse=True)
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
        expected = defaultdict(list)
        for node in reference_nodes:
            if len(node.qargs) not in (1, 2):
                continue
            line = format_node_line(node)
            for q in op_node_signature(node)[2]:
                expected[q].append(line)
        if dict(traces) != dict(expected):
            raise ValueError("Gate identities, multiplicities, or per-qubit order changed.")
    return positions


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
            singles[i], config["average_single_gate_time"], config["t_switch"]
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
        singles[-1], config["average_single_gate_time"], config["t_switch"]
    )
    plan.events.extend(("gate", line) for line in lines)
    _return_plan(plan, homes, config)
    return plan


def _sequential_candidate(ops, homes, config):
    """Home-return baseline with naive_dag's next-2Q-gate mover lookahead."""
    from .scheduling import single_qubit_layer_time
    plan = _Plan(dict(homes), [], [])
    two_indices = [i for i, n in enumerate(ops) if len(n.qargs) == 2]
    next_pairs = {}
    for a, b in zip(two_indices, two_indices[1:]):
        next_pairs[a] = set(op_node_signature(ops[b])[2])
    i = 0
    while i < len(ops):
        node = ops[i]
        if len(node.qargs) != 2:
            block = []
            while i < len(ops) and len(ops[i].qargs) != 2:
                block.append(ops[i])
                i += 1
            lines, _ = single_qubit_layer_time(
                block, config["average_single_gate_time"], config["t_switch"]
            )
            plan.events.extend(("gate", line) for line in lines)
            continue
        pair = op_node_signature(node)[2]
        if plan.ids and plan.ids[0] not in pair:
            _return_plan(plan, homes, config)
        following = next_pairs.get(i, set())
        mover = plan.ids[0] if plan.ids else (pair[1] if pair[1] in following else pair[0])
        _execute_group(plan, [node], [mover], config, reuse=bool(plan.ids))
        if mover not in following:
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
        if len(ops[i].qargs) == 2:
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
            while i < len(ops) and len(ops[i].qargs) != 2:
                block.append(ops[i])
                i += 1
            lines, _ = single_qubit_layer_time(
                block, config["average_single_gate_time"], config["t_switch"]
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
    """Select a validated complete schedule with common homes and timing.

    Always include a reset-only plan and a safe sequential baseline. Include
    the actual naive_dag trajectory when it satisfies the same rules.
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
    factories = {
        "reset_only": lambda: _layered_candidate(layers, singles, homes, config, False),
        "reuse": lambda: _layered_candidate(layers, singles, homes, config, True),
        "sequential_home": lambda: _sequential_candidate(reference_nodes, homes, config),
        "naive_dag": lambda: _legacy_baseline(reference_nodes, homes, config),
    }
    candidates = {}
    rejected = {}
    for name, build in factories.items():
        try:
            plan = build()
            validate_schedule(plan.events, homes, config, reference_nodes)
            candidates[name] = (schedule_duration(plan.events, config), plan)
        except (ValueError, RuntimeError, IndexError) as exc:
            rejected[name] = str(exc)
    if "sequential_home" not in candidates:
        raise ValueError(f"Sequential home baseline failed validation: {rejected.get('sequential_home')}")
    selected = min(candidates, key=lambda name: candidates[name][0])
    duration, plan = candidates[selected]
    return CircuitSchedule(
        plan.events, duration, selected,
        {name: time for name, (time, _) in candidates.items()}, rejected, plan.reused_groups,
    )
