from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import op_node_signature
from .grid import GridNode, Qubit, generate_grid, move_qubit, place_qubit
from .scheduling import _format_node_line, collect_single_qubit_gate_block

MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent
_CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _slm(point):
    return point[0] % 2 == 0 and point[1] % 2 == 0


def _neighbors(point):
    return [(point[0] + dr, point[1] + dc) for dr, dc in _CARDINAL]


def _highway_segment(start, end):
    """Check the entire segment: horizontal on odd rows, vertical on odd columns."""
    if _slm(start) or _slm(end):
        return False
    return ((start[0] == end[0] and start[0] % 2 == 1)
            or (start[1] == end[1] and start[1] % 2 == 1))


def _parameters(config):
    return (
        config['rydberg_radius'].to('meter').magnitude / 2,
        config['max_velocity'].to('meter/second').magnitude,
        config['max_acceleration'].to('meter/second^2').magnitude,
        config['transfer_SLM_AOD'].to('seconds').magnitude,
    )


@lru_cache(maxsize=8192)
def _motion_seconds(distance, velocity, acceleration):
    if distance == 0:
        return 0.0
    ramp = velocity * velocity / acceleration
    if distance > ramp:
        return 2 * velocity / acceleration + (distance - ramp) / velocity
    return 2 * math.sqrt(distance / acceleration)


def _path_seconds(points, parameters):
    spacing, velocity, acceleration, transfer = parameters
    total = 0.0
    for a, b in zip(points, points[1:]):
        if a == b:
            continue
        total += _motion_seconds(math.hypot(b[0] - a[0], b[1] - a[1]) * spacing,
                                 velocity, acceleration)
        if _slm(a) or _slm(b):
            total += transfer
    return total


def _compact(points):
    """Merge same-direction highway pieces; never merge a turn or transfer."""
    result = []
    for p in points:
        if result and result[-1] == p:
            continue
        if len(result) >= 2:
            a, b = result[-2:]
            same_direction = ((b[0] - a[0]) * (p[0] - b[0]) > 0 and a[1] == b[1] == p[1]
                              or (b[1] - a[1]) * (p[1] - b[1]) > 0 and a[0] == b[0] == p[0])
            if same_direction and _highway_segment(a, b) and _highway_segment(b, p):
                result[-1] = p
                continue
        result.append(p)
    return tuple(result)


def _route_key(points, parameters):
    return _path_seconds(points, parameters), len(points), points


@lru_cache(maxsize=32768)
def _highway_route(start, end, parameters):
    """Route AOD coordinates through adjacent odd-odd intersections.

    Coordinates outside the SLM rectangle remain legal highway coordinates;
    they are never used as indices into the finite SLM grid.
    """
    if _slm(start) or _slm(end):
        raise ValueError('Highway endpoints must be AOD positions.')
    if start == end:
        return (start,)

    def intersections(p):
        r, c = p
        if r % 2 and c % 2:
            return [p]
        if r % 2:
            return [(r, c - 1), (r, c + 1)]
        return [(r - 1, c), (r + 1, c)]

    candidates = []
    if _highway_segment(start, end):
        candidates.append((start, end))
    for a in intersections(start):
        for b in intersections(end):
            for corner in ((a[0], b[1]), (b[0], a[1])):
                candidates.append(_compact((start, a, corner, b, end)))
    return min(candidates, key=lambda path: _route_key(path, parameters))


@lru_cache(maxsize=32768)
def _interaction_route(start, target, parameters):
    if not _slm(target):
        raise ValueError('The interaction partner must remain in an SLM trap.')
    candidates = []
    for first in _neighbors(start) if _slm(start) else [start]:
        for end in _neighbors(target):
            route = _highway_route(first, end, parameters)
            candidates.append((start,) + route if _slm(start) else route)
    return min(candidates, key=lambda path: _route_key(path, parameters))


@lru_cache(maxsize=32768)
def _unload_route(start, destination, parameters):
    if _slm(start) or not _slm(destination):
        raise ValueError('Unloading requires an AOD start and an SLM destination.')
    return min((_highway_route(start, end, parameters) + (destination,)
                for end in _neighbors(destination)),
               key=lambda path: _route_key(path, parameters))


def gate_qubit_ids(ops: list[DAGOpNode], gate_index: int) -> list[int]:
    _, _, ids = op_node_signature(ops[gate_index])
    if len(ids) > 2:
        raise ValueError('This method does not support 3+ qubit gates')
    return ids


def gate_qubits(ops, gate_index, qubits):
    ids = gate_qubit_ids(ops, gate_index)
    if len(ids) != 2 or ops[gate_index].op.name == 'swap':
        return [None, None]
    mapping = {q.id: q for q in qubits}
    return [mapping[q] for q in ids]


def find_next_two_qubit_gate(ops, start_index):
    return next((i for i in range(start_index + 1, len(ops))
                 if len(ops[i].qargs) == 2 and ops[i].op.name != 'swap'), None)


def _future_pairs(ops, gate_index, mover, config):
    """Bound the rollout by both all upcoming 2Q gates and uses of this atom."""
    max_gates = int(config.get('relocation_lookahead_gates', 8))
    max_uses = int(config.get('relocation_lookahead_uses', 2))
    if max_gates < 1 or max_uses < 1:
        raise ValueError('Relocation lookahead limits must be positive.')
    pairs, uses = [], 0
    for node in ops[gate_index + 1:]:
        if len(node.qargs) != 2 or node.op.name == 'swap':
            continue
        pair = tuple(op_node_signature(node)[2])
        pairs.append(pair)
        uses += mover in pair
        if len(pairs) >= max_gates or uses >= max_uses:
            break
    return pairs


def _choose_mover(pair, following, positions):
    loaded = [q for q, p in positions.items() if not _slm(p)]
    if len(loaded) > 1 or loaded and loaded[0] not in pair:
        raise ValueError('Return the existing AOD atom before loading a different mover.')
    if loaded:
        return loaded[0]
    return pair[1] if pair[1] in following else pair[0]


def _rollout_cost(positions, pairs, parameters):
    """Simulate intervening gates on private state with previous-home returns.

    A hypothetical closing unload gives all options an unloaded terminal state.
    Immediate reuse is mandatory inside the horizon. No recursive relocation.
    """
    positions = dict(positions)
    homes = dict(positions)
    total = 0.0
    for i, pair in enumerate(pairs):
        following = pairs[i + 1] if i + 1 < len(pairs) else ()
        mover = _choose_mover(pair, following, positions)
        partner = pair[1] if mover == pair[0] else pair[0]
        route = _interaction_route(positions[mover], positions[partner], parameters)
        total += _path_seconds(route, parameters)
        positions[mover] = route[-1]
        if mover not in following:
            route = _unload_route(positions[mover], homes[mover], parameters)
            total += _path_seconds(route, parameters)
            positions[mover] = route[-1]
    return total


def _choose_unload(mover, positions, previous_home, grid, pairs, parameters, config):
    occupied = {p for q, p in positions.items() if q != mover}
    vacant = [(r, c) for r in range(0, len(grid), 2)
              for c in range(0, len(grid[0]), 2) if (r, c) not in occupied]
    if previous_home not in vacant:
        raise ValueError('The moving atom\'s previous home is unexpectedly occupied.')
    if not config.get('relocation_enabled', True) or len(vacant) == 1:
        return _unload_route(positions[mover], previous_home, parameters)

    limit = int(config.get('relocation_candidates', 8))
    if limit < 1:
        raise ValueError('relocation_candidates must be positive.')
    routes = {p: _unload_route(positions[mover], p, parameters) for p in vacant}
    nearby = sorted(vacant, key=lambda p: _route_key(routes[p], parameters))
    partners = [positions[b if a == mover else a] for a, b in pairs if mover in (a, b)]
    rankings = [nearby] + [sorted(vacant, key=lambda p: (abs(p[0] - t[0]) + abs(p[1] - t[1]), p))
                          for t in partners]
    # Always include the previous home. Round-robin nearby and future-partner sites.
    candidates = [previous_home]
    for rank in range(len(vacant)):
        for ranking in rankings:
            if len(candidates) >= limit:
                break
            p = ranking[rank]
            if p not in candidates:
                candidates.append(p)
        if len(candidates) >= limit:
            break

    def cost(p):
        future_positions = dict(positions)
        future_positions[mover] = p
        return (_path_seconds(routes[p], parameters)
                + _rollout_cost(future_positions, pairs, parameters))

    selected, best = previous_home, cost(previous_home)
    for p in candidates[1:]:
        candidate_cost = cost(p)
        if candidate_cost < best - 1e-12:
            selected, best = p, candidate_cost
    return routes[selected]


def _set_position(qubit, position, grid, config):
    if _slm(position):
        r, c = position
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            raise ValueError('Unloading destination is outside the SLM grid.')
        node = grid[r][c]
    else:
        r, c = position
        spacing = config['rydberg_radius'] / 2
        node = GridNode(x=c * spacing, y=r * spacing, row=r, col=c)
    move_qubit(qubit, node)


def best_path_for_gate(ops, gate_index, qubits, grid, config, T):
    """Single-mover parity routing with mandatory next-gate retention.

    The legacy public signature and absolute-position move events are preserved.
    Relocation is considered only when this atom must leave the AOD.
    """
    q1, q2 = gate_qubits(ops, gate_index, qubits)
    if q1 is None:
        return 0 * config['t_switch'], T, []
    mapping = {q.id: q for q in qubits}
    positions = {q.id: q.grid_position() for q in qubits}
    pair = tuple(gate_qubit_ids(ops, gate_index))
    next_index = find_next_two_qubit_gate(ops, gate_index)
    following = tuple(gate_qubit_ids(ops, next_index)) if next_index is not None else ()
    mover = _choose_mover(pair, following, positions)
    partner = pair[1] if mover == pair[0] else pair[0]
    atom = mapping[mover]
    if _slm(positions[mover]):
        atom._naive_home = positions[mover]
    elif not hasattr(atom, '_naive_home'):
        raise ValueError('A retained AOD atom needs its previous SLM home.')
    parameters = _parameters(config)
    route = _interaction_route(positions[mover], positions[partner], parameters)
    events = [('move', mover, a, b) for a, b in zip(route, route[1:])]
    positions[mover] = route[-1]
    if sum(abs(a - b) for a, b in zip(route[-1], positions[partner])) != 1:
        raise ValueError('Gate operands must be one grid step apart.')
    events.append(('gate', _format_node_line(ops[gate_index])))
    lines, _, pulse_counts = collect_single_qubit_gate_block(ops, gate_index + 1)
    events.extend(('gate', line) for line in lines)
    seconds = _path_seconds(route, parameters)
    if mover not in following:
        pairs = _future_pairs(ops, gate_index, mover, config)
        returning = _choose_unload(mover, positions, atom._naive_home, grid,
                                   pairs, parameters, config)
        events.extend(('move', mover, a, b) for a, b in zip(returning, returning[1:]))
        seconds += _path_seconds(returning, parameters)
        positions[mover] = returning[-1]
        atom._naive_home = returning[-1]
    _set_position(atom, positions[mover], grid, config)
    duration = (seconds * config['t_switch'].to('seconds').units
                + config['average_two_gate_time'] + config['t_switch']
                + sum(pulse_counts) * (config['average_single_gate_time'] + config['t_switch']))
    return duration, T + len(events), events



def validate_schedule(events, initial, config, reference_nodes):
    """Replay one-AOD legality and SWAP-filtered wire order; allow changed homes."""
    positions = dict(initial)
    rows, cols = config['dimensions']
    if len(set(initial.values())) != len(initial) or any(
        not _slm(p) or not (0 <= p[0] < 2 * rows - 1 and 0 <= p[1] < 2 * cols - 1)
        for p in initial.values()
    ):
        raise ValueError('Initial positions must be unique in-bounds SLM traps.')
    active = None
    traces = defaultdict(list)
    next_pairs = [()] * len(events)
    following = ()
    for index in range(len(events) - 1, -1, -1):
        next_pairs[index] = following
        event = events[index]
        if event[0] == 'gate':
            for statement in reversed(event[1].split(';')):
                ids = tuple(map(int, re.findall(r'q\[(\d+)\]', statement)))
                if len(ids) == 2:
                    following = ids
    for index, event in enumerate(events):
        if event[0] == 'gate':
            used = set()
            for statement in event[1].split(';'):
                statement = statement.strip()
                if not statement:
                    continue
                ids = list(map(int, re.findall(r'q\[(\d+)\]', statement)))
                if (not ids or len(set(ids)) != len(ids) or used.intersection(ids)
                        or not set(ids) <= positions.keys() or statement.startswith('swap ')):
                    raise ValueError('Invalid or overlapping gate operands, or an emitted SWAP.')
                used.update(ids)
                if len(ids) == 2:
                    if active not in ids or sum(abs(a-b) for a,b in zip(positions[ids[0]], positions[ids[1]])) != 1:
                        raise ValueError('A 2Q gate needs one AOD atom adjacent to its SLM partner.')
                elif len(ids) != 1:
                    raise ValueError('Only one- and two-qubit gates are supported.')
                for q in ids:
                    traces[q].append(statement + ';')
            continue
        _, q, a, b = event
        if q not in positions or positions[q] != a or a == b:
            raise ValueError('Movement continuity violation or no-op event.')
        if _slm(a):
            if active is not None or _slm(b):
                raise ValueError('Load requires an empty AOD and an AOD destination.')
            active = q
        elif active != q:
            raise ValueError('Only the active atom may move.')
        if _slm(a) or _slm(b):
            if sum(abs(x-y) for x,y in zip(a,b)) != 1:
                raise ValueError('Transfers must be one cardinal step.')
        elif not _highway_segment(a,b):
            raise ValueError('Movement crosses an SLM site or leaves the highway.')
        if _slm(b):
            if not (0 <= b[0] < 2 * rows - 1 and 0 <= b[1] < 2 * cols - 1):
                raise ValueError('Unload outside the SLM grid.')
            if q in next_pairs[index]:
                raise ValueError('An atom needed by the next 2Q gate must remain loaded.')
            active = None
        if any(other != q and p == b for other,p in positions.items()):
            raise ValueError('Movement ends at an occupied position.')
        positions[q] = b
    if active is not None:
        raise ValueError('The completed schedule must unload its remaining AOD atom.')
    expected = defaultdict(list)
    for node in reference_nodes:
        if node.op.name in ('swap', 'barrier'):
            continue
        if len(node.qargs) not in (1, 2):
            raise ValueError('Unsupported reference operation.')
        for q in op_node_signature(node)[2]:
            expected[q].append(_format_node_line(node))
    if dict(traces) != dict(expected):
        raise ValueError('Gate identities, measurement destinations, or per-qubit order changed.')
    return positions


@dataclass
class CircuitSchedule:
    events: list[ScheduleEvent]
    duration: object
    selected: str
    candidate_times: dict
    final_positions: dict


def schedule_circuit(ops, qubits, config):
    """Choose the faster complete valid plan, preserving the caller's placement.

    Both candidates retain immediate reuse. They differ only in whether a
    required unload returns to the previous home or uses relocation lookahead.
    """
    ops = [n for n in ops if n.op.name not in ('swap', 'barrier')]
    initial = {q.id: q.grid_position() for q in qubits}
    plans = {}
    modes = [('previous_home', False)]
    if config.get('relocation_enabled', True):
        modes.append(('relocation', True))
    for name, enabled in modes:
        grid = generate_grid(config['dimensions'], config['rydberg_radius'])
        placed = [place_qubit(grid, *initial[q], q) for q in sorted(initial)]
        cfg = dict(config, relocation_enabled=enabled)
        lines, first, counts = collect_single_qubit_gate_block(ops, 0)
        events = [('gate', line) for line in lines]
        duration = sum(counts) * (config['average_single_gate_time'] + config['t_switch'])
        i = first
        while i < len(ops):
            delta, _, emitted = best_path_for_gate(ops, i, placed, grid, cfg, len(events))
            duration += delta
            events.extend(emitted)
            following = find_next_two_qubit_gate(ops, i)
            if following is None:
                break
            i = following
        ending = validate_schedule(events, initial, config, ops)
        plans[name] = (duration, events, ending)
    selected = min(plans, key=lambda name: plans[name][0])
    duration, events, ending = plans[selected]
    return CircuitSchedule(events, duration, selected,
                           {name: plan[0] for name, plan in plans.items()}, ending)
