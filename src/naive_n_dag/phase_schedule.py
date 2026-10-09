"""Route a commuting CZ block as one displacement tour and one rigid load."""
from collections import defaultdict, deque

from .dag_helper import _is_reorderable_cz, _validate_cz_reordering, op_node_signature
from .dynamics import (
    CircuitSchedule, _Plan, _check_layer, _execute_group, _fits_same_aod,
    _max_grid_spans, _motion_parameters, _movement_vector,
    _order_group_displacements, _optimize_visit_sides, _preferred_mover, _reset_layer,
    _return_plan, _reuse_group_allowed, _start_, _translate,
    schedule_duration, validate_schedule,
)
from .precompile import precompile_commuting_blocks
from .scheduling import single_qubit_layer_time


def _phase_movers(nodes, config):
    """Choose a consistent moving bipartition, honoring all mover preferences."""
    pairs = [op_node_signature(node)[2] for node in nodes]
    adjacency = defaultdict(list)
    required = {}
    for pair in pairs:
        a, b = pair
        adjacency[a].append(b)
        adjacency[b].append(a)
        preferred = _preferred_mover(pair, config)
        if preferred is not None:
            for q in pair:
                moving = q == preferred
                if q in required and required[q] != moving:
                    raise ValueError("Phase has conflicting mover preferences.")
                required[q] = moving

    movers = set()
    seen = set()
    for root in adjacency:
        if root in seen:
            continue
        sides = {root: 0}
        queue = deque([root])
        while queue:
            q = queue.popleft()
            for partner in adjacency[q]:
                if partner not in sides:
                    sides[partner] = 1 - sides[q]
                    queue.append(partner)
                elif sides[partner] == sides[q]:
                    raise ValueError("A phase-wide rigid load requires a bipartite CZ block.")
        choices = {sides[q] if moving else 1 - sides[q]
                   for q, moving in required.items() if q in sides}
        if len(choices) > 1:
            raise ValueError("Phase bipartition cannot satisfy mover preferences.")
        moving_side = next(iter(choices), 0)
        movers.update(q for q, side in sides.items() if side == moving_side)
        seen.update(sides)
    return [a if a in movers else b for a, b in pairs]


def _phase_buckets(nodes, movers, homes):
    buckets = defaultdict(list)
    for node, mover in zip(nodes, movers):
        pair = op_node_signature(node)[2]
        partner = pair[1] if mover == pair[0] else pair[0]
        buckets[_movement_vector(homes[mover], homes[partner])].append((node, mover))
    # Distinct fixed-offset bipartite edges cannot overlap except for repeated
    # occurrences of the same edge. Keep such occurrences in separate pulses.
    visits = []
    for bucket in buckets.values():
        matchings = []
        for node, mover in bucket:
            used = set(op_node_signature(node)[2])
            for group, group_movers, occupied in matchings:
                if not used & occupied:
                    group.append(node)
                    group_movers.append(mover)
                    occupied.update(used)
                    break
            else:
                matchings.append(([node], [mover], set(used)))
        visits.append([(group, group_movers) for group, group_movers, _ in matchings])
    return list(buckets), visits


def _route_phase(nodes, homes, config):
    # Phase-wide side choices must not be replaced by per-group optimization.
    # The greedy executor emits already-adjacent gates without moving them.
    execution_config = dict(config, _side_ordering="greedy")
    movers = _phase_movers(nodes, config)
    ids = sorted(set(movers))
    positions = [homes[q] for q in ids]
    if not config.get("parallel", False) and len(ids) > 1:
        raise ValueError("Phase load requires parallel movement.")
    if not _fits_same_aod(positions[1:], positions[0], *_max_grid_spans(config["max_dimension"])):
        raise ValueError("Complete phase load exceeds the configured AOD span.")
    vectors, buckets = _phase_buckets(nodes, movers, homes)
    order = _order_group_displacements(vectors, config)

    def route(candidate_order, optimize_sides=False):
        visits = [visit for index in candidate_order for visit in buckets[index]]
        next_uses = [{} for _ in visits]
        future = {}
        for i in reversed(range(len(visits))):
            next_uses[i] = dict(future)
            future.update({q: i for q in visits[i][1]})
        for i, (_, active_movers) in enumerate(visits):
            if not _reuse_group_allowed(ids, active_movers, config.get("T_reuse", 0), i, next_uses[i]):
                raise ValueError("Complete phase retention exceeds the configured T_reuse horizon.")

        plan = _Plan(dict(homes), list(ids), [])

        def translate(step):
            current = [plan.positions[q] for q in plan.ids]
            _translate(plan.ids, current, step, plan.events, [])
            plan.positions.update(zip(plan.ids, current))

        if optimize_sides:
            tour = _optimize_visit_sides(
                [vectors[index] for index in candidate_order for _ in buckets[index]], config,
            )
            translate(tour.load_step)
            for (group, active_movers), steps in zip(visits, tour.routes):
                for step in steps:
                    translate(step)
                _check_layer(group, homes)
                # Every gate is already adjacent on its planned side. The
                # existing executor checks/emits gates without choosing a move.
                _execute_group(plan, group, active_movers, homes, execution_config, reuse=True)
            for step in tour.return_steps:
                translate(step)
            if plan.positions != homes:
                raise ValueError("Side-optimized tour failed to restore original homes.")
            plan.ids = []
        else:
            # Use greedy sides only when explicitly configured.
            loaded_positions = list(positions)
            _start_([vectors[candidate_order[0]]] * len(ids), [], ids, loaded_positions, plan.events)
            plan.positions.update(zip(ids, loaded_positions))
            for group, active_movers in visits:
                _check_layer(group, homes)
                _execute_group(plan, group, active_movers, homes, execution_config, reuse=True)
            _return_plan(plan, homes, config)
        plan.reused_groups = max(0, len(visits) - 1)
        validate_schedule(plan.events, homes, config, [node for group, _ in visits for node in group])
        return plan, [group for group, _ in visits]

    # Compare tour directions within the configured side policy, never policies.
    optimize_sides = config.get("_side_ordering") != "greedy"
    candidates = []
    failures = []
    orders = [order, list(reversed(order))] if len(order) > 1 else [order]
    for candidate_order in orders:
        try:
            candidates.append(route(candidate_order, optimize_sides))
        except ValueError as exc:
            failures.append(str(exc))
    if not candidates:
        raise ValueError(failures[0])
    return min(candidates, key=lambda candidate: schedule_duration(candidate[0].events, config))


def schedule_phase_tours(reference_nodes, qubits, config):
    """Return a replay-validated phase candidate and its actual layer/context order.

    Each compatible commuting CZ block gets one home-to-home tour over unique
    displacement targets. Unsupported full loads reject this optional candidate;
    the caller retains ordinary original-layer and packed-layer schedules.
    """
    original = list(reference_nodes)
    homes = {q.id: q.grid_position() for q in qubits}
    config = dict(config, _motion_parameters=_motion_parameters(config), _ancilla_qubits=frozenset(
        q for node in original if node.op.name == "reset" for q in op_node_signature(node)[2]
    ))
    blocks, contexts, _ = precompile_commuting_blocks(original, homes, config)
    plan = _Plan(dict(homes), [], [])
    layers, singles, reference = [], [[]], []

    def emit_context(nodes):
        lines, _ = single_qubit_layer_time(
            nodes, config["average_single_gate_time"], config["t_switch"], config.get("measurement_time"),
        )
        plan.events.extend(("gate", line) for line in lines)
        singles[-1].extend(nodes)
        reference.extend(nodes)

    for i, block in enumerate(blocks):
        emit_context(contexts[i])
        if block and all(_is_reorderable_cz(node) for node in block):
            phase, visits = _route_phase(block, homes, config)
            plan.events.extend(phase.events)
            plan.reused_groups += phase.reused_groups
        else:
            _reset_layer(plan, block, homes, config)
            _return_plan(plan, homes, config)
            visits = [block]
        for visit in visits:
            layers.append(visit)
            singles.append([])
            reference.extend(visit)
    emit_context(contexts[-1])
    _validate_cz_reordering(original, reference)
    validate_schedule(plan.events, homes, config, reference)
    duration = schedule_duration(plan.events, config)
    result = CircuitSchedule(plan.events, duration, "phase_tour", {"phase_tour": duration}, {}, plan.reused_groups)
    return result, (layers, singles, reference)
