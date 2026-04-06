from __future__ import annotations

import math
from collections import deque
from typing import Any

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import format_gate_line, op_node_signature
from .grid import Qubit

# TODO: helper file that has dynamics equations for solving for time using the parameters in the JSON file between.
# If there is an atom there you must first offload it, then load the site. If there is nothing there then we can load it.

MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent


def _max_grid_spans(max_dimension: list[int]) -> tuple[int, int]:
    """Convert AOD atom dimensions into max grid-index spans on even-even lattice sites."""
    max_rows, max_cols = max_dimension
    return 2 * (max_rows - 1), 2 * (max_cols - 1)


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


def _reduce_vectors_for_start(vectors: list[tuple[int, int]], move: tuple[int, int]) -> list[tuple[int, int]]:
    """When starting from (0,0), convert vectors into successive differences."""
    if move != (0, 0) or not vectors:
        return vectors[:]

    reduced = [vectors[0]]
    for idx in range(1, len(vectors)):
        prev_x, prev_y = vectors[idx - 1]
        cur_x, cur_y = vectors[idx]
        reduced.append((cur_x - prev_x, cur_y - prev_y))
    return reduced


def _start_(
    moving_vectors: list[tuple[int, int]],
    move: list[int],
    moving_ids: list[int],
    origin_positions: list[tuple[int, int]],
    event_log: list[ScheduleEvent],
) -> list[tuple[int, int]]:
    """Apply initial vector reduction and optional parity-prep move for all movers."""
    reduced_vectors = _reduce_vectors_for_start(moving_vectors, (move[0], move[1]))
    #Assume that they start in original position, even-even
    dx, dy = reduced_vectors[0]
    next_dx, next_dy = 0, 0
    for idx, (vec_x,vec_y) in enumerate(reduced_vectors):
        if (vec_x,vec_y) != (0,0):
            next_dx, next_dy = reduced_vectors[idx]
            break
    if (origin_positions[0][0] % 2 == 0) and (origin_positions[0][1] % 2 == 0):
        # If the first move is in a grid position
        if dx == 0:
            if dy == 2 and abs(next_dx) > 2 and next_dy == 0:
                move[1] +=1
            elif dy == -2 and abs(next_dx) > 2 and next_dy == 0:
                move[1] -=1
            elif next_dx >= 2:
                move[0] +=1
            else:
                move[0] -=1
        elif dy == 0:
            if dx == 2 and abs(next_dy) > 2 and next_dx == 0:
                move[0] +=1
            elif dx == -2 and abs(next_dy) > 2 and next_dx == 0:
                move[0] -=1
            elif next_dy >= 2:
                move[1] +=1
            else:
                move[1] -=1
        else:
            if next_dx == 2:
                move[0] +=1
            elif next_dx == -2:
                move[0] -=1
            elif next_dy == 2:
                move[1] +=1
            elif next_dy == -2:
                move[1] -=1
            elif abs(dx) > abs(dy):
                if dx > 0:
                    move[0] +=1
                else:
                    move[0] -=1
            else:
                if dy > 0:
                    move[1] +=1
                else:
                    move[1] -=1
        
    for idx, (vec_x, vec_y) in enumerate(reduced_vectors):
        base_r, base_c = origin_positions[idx]
        start = (base_r, base_c)
        end = (base_r + move[0], base_c + move[1])
        origin_positions[idx] = end
        reduced_vectors[idx] = (vec_x - move[0], vec_y - move[1]) #shift the vector back to the origin
        event_log.append(("move", moving_ids[idx], start, end))

    return reduced_vectors


def _shuttle_(
    moving_vectors: list[tuple[int, int]],
    move: list[int],
    moving_ids: list[int],
    moving_group_positions: list[tuple[int, int]],
    gate_nodes: list[DAGOpNode],
    event_log: list[ScheduleEvent],
) -> None:
    """Route on highway-constrained unit steps and emit gates when target vector reaches magnitude 1."""

    def _nonzero_indices(vectors: list[tuple[int, int]]) -> list[int]:
        return [idx for idx, (x, y) in enumerate(vectors) if (x, y) != (0, 0)]

    def _is_unit_axis(step: tuple[int, int]) -> bool:
        sx, sy = step
        return abs(sx) + abs(sy) == 1

    def _is_highway_legal(cur_move: tuple[int, int], step: tuple[int, int]) -> bool:
        sx, sy = step
        if not _is_unit_axis(step):
            return False
        cur_x, cur_y = cur_move
        if sx != 0:
            return cur_y % 2 != 0
        return cur_x % 2 != 0

    def _emit_group_move(step: tuple[int, int]) -> None:
        sx, sy = step
        if not _is_unit_axis(step):
            raise ValueError(f"Non-axis unit move requested in shuttle: {step}")
        for idx, moving_id in enumerate(moving_ids):
            start = moving_group_positions[idx]
            end = (start[0] + sx, start[1] + sy)
            event_log.append(("move", moving_id, start, end))
            moving_group_positions[idx] = end
        move[0] += sx
        move[1] += sy
        for idx, (vec_x, vec_y) in enumerate(moving_vectors):
            moving_vectors[idx] = (vec_x - sx, vec_y - sy)

    def _prefer_axis_step(
        vec: tuple[int, int],
        cur_move: tuple[int, int],
        preferred_axis: str | None = None,
    ) -> tuple[int, int] | None:
        vx, vy = vec
        options: list[tuple[int, int]] = []
        if vx != 0:
            options.append((1 if vx > 0 else -1, 0))
        if vy != 0:
            options.append((0, 1 if vy > 0 else -1))
        if preferred_axis == "x":
            options.sort(key=lambda s: 0 if s[0] != 0 else 1)
        elif preferred_axis == "y":
            options.sort(key=lambda s: 0 if s[1] != 0 else 1)
        else:
            options.sort(key=lambda s: 0 if (abs(vx) >= abs(vy) and s[0] != 0) or (abs(vy) > abs(vx) and s[1] != 0) else 1)
        for step in options:
            if _is_highway_legal(cur_move, step):
                return step
        return None

    def _legal_steps(cur_move: tuple[int, int]) -> list[tuple[int, int]]:
        cur_x, cur_y = cur_move
        steps: list[tuple[int, int]] = []
        if cur_y % 2 != 0:
            steps.extend([(1, 0), (-1, 0)])
        if cur_x % 2 != 0:
            steps.extend([(0, 1), (0, -1)])
        return steps

    def _plan_step_toward_unit(
        cur_move: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Return first step of a shortest legal path from cur_move to any state with |target-state|<=1."""
        cx, cy = cur_move
        tx, ty = target
        if math.hypot(tx - cx, ty - cy) <= 1:
            return None

        margin = max(6, abs(tx - cx) + abs(ty - cy) + 4)
        min_x, max_x = min(cx, tx) - margin, max(cx, tx) + margin
        min_y, max_y = min(cy, ty) - margin, max(cy, ty) + margin

        start = (cx, cy)
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        goal: tuple[int, int] | None = None

        while queue:
            x, y = queue.popleft()
            if math.hypot(tx - x, ty - y) <= 1:
                goal = (x, y)
                break
            for sx, sy in _legal_steps((x, y)):
                nx, ny = x + sx, y + sy
                if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                    continue
                nxt = (nx, ny)
                if nxt in parent:
                    continue
                parent[nxt] = (x, y)
                queue.append(nxt)

        if goal is None:
            return None

        cur = goal
        while parent[cur] != start and parent[cur] is not None:
            cur = parent[cur]
        if parent[cur] is None:
            return None
        return cur[0] - cx, cur[1] - cy

    emitted_gate_indices: set[int] = set()

    for idx in range(len(moving_vectors)):
        local_iter = 0
        while math.hypot(moving_vectors[idx][0], moving_vectors[idx][1]) > 1:
            local_iter += 1
            if local_iter > 10000:
                raise RuntimeError(
                    f"Shuttle failed to converge for gate index {idx}. "
                    f"move={tuple(move)} vec={moving_vectors[idx]}"
                )

            cur_vec = moving_vectors[idx]
            next_idx = next(
                (j for j in range(idx + 1, len(moving_vectors)) if moving_vectors[j] != (0, 0)),
                None,
            )
            step: tuple[int, int] | None = None

            if next_idx is not None:
                next_vec = moving_vectors[next_idx]
                delta_x = next_vec[0] - cur_vec[0]
                delta_y = next_vec[1] - cur_vec[1]
                if (delta_x, delta_y) in {(2, 0), (-2, 0), (0, 2), (0, -2)}:
                    if abs(cur_vec[0]) == 2 and cur_vec[1] == 0:
                        candidate = (1 if cur_vec[0] > 0 else -1, 0)
                        if _is_highway_legal((move[0], move[1]), candidate):
                            step = candidate
                    elif abs(cur_vec[1]) == 2 and cur_vec[0] == 0:
                        candidate = (0, 1 if cur_vec[1] > 0 else -1)
                        if _is_highway_legal((move[0], move[1]), candidate):
                            step = candidate

            if step is None:
                step = _prefer_axis_step(cur_vec, (move[0], move[1]))

            if step is None:
                target = (move[0] + cur_vec[0], move[1] + cur_vec[1])
                step = _plan_step_toward_unit((move[0], move[1]), target)
                if step is None:
                    raise RuntimeError(
                        f"Shuttle could not find legal progress step for gate index {idx} "
                        f"at move={tuple(move)} target={target}."
                    )

            _emit_group_move(step)

            if idx == len(moving_vectors) - 1 and local_iter % 3 == 0:
                # Last target: enforce batches of at most 3 movement updates before re-evaluating.
                pass

        if idx not in emitted_gate_indices:
            gate_name, gate_params, qubit_ids = op_node_signature(gate_nodes[idx])
            event_log.append(("gate", format_gate_line(gate_name, gate_params, qubit_ids)))
            emitted_gate_indices.add(idx)
            moving_vectors[idx] = (0, 0)


def _return_(
    moving_ids: list[int],
    current_positions: list[tuple[int, int]],
    home_positions: list[tuple[int, int]],
    event_log: list[ScheduleEvent],
) -> None:
    """Log synchronized unit-step return moves for all moving qubits."""
    if not moving_ids:
        return
    if len(moving_ids) != len(current_positions) or len(moving_ids) != len(home_positions):
        raise ValueError("Return path inputs must have matching lengths.")

    current = current_positions[:]
    ref_cur_r, ref_cur_c = current[0]
    ref_home_r, ref_home_c = home_positions[0]

    while (ref_cur_r, ref_cur_c) != (ref_home_r, ref_home_c):
        step_r = 0
        step_c = 0
        if ref_cur_r != ref_home_r:
            step_r = -1 if ref_cur_r > ref_home_r else 1
        elif ref_cur_c != ref_home_c:
            step_c = -1 if ref_cur_c > ref_home_c else 1

        for idx, moving_id in enumerate(moving_ids):
            start = current[idx]
            end = (start[0] + step_r, start[1] + step_c)
            event_log.append(("move", moving_id, start, end))
            current[idx] = end

        ref_cur_r += step_r
        ref_cur_c += step_c

    for idx, home in enumerate(home_positions):
        if current[idx] != home:
            raise RuntimeError("Synchronized return path did not land all qubits at home positions.")



def _is_opposite_direction(
    group_vectors: list[tuple[int, int]],
    candidate_vector: tuple[int, int],
    alignment_conc: float,
) -> bool:
    """Return True when candidate movement is below the configured alignment threshold."""
    if not group_vectors:
        return False
    return _vector_alignment_score(group_vectors, candidate_vector) < alignment_conc


def best_path_for_layer(
    layer_nodes: list[DAGOpNode],
    qubits: list[Qubit],
    config: dict[str, Any],
    event_log: list[ScheduleEvent],
) -> tuple[int, Any]:
    """Group 2Q gates by AOD fit and append grouped gate steps into ``event_log`` in place.

    One atom per gate is selected as the moving candidate based on AOD fit and movement
    vector compatibility with existing atoms already assigned to a moving group.
    Returns the number of emitted gate timesteps and time contribution.
    """
    if not layer_nodes:
        zero_time = 0 * (config["average_two_gate_time"] + config["t_switch"])
        return 0, zero_time

    qubit_map = {q.id: q for q in qubits}
    max_row_span, max_col_span = _max_grid_spans(config["max_dimension"])
    alignment_conc = float(config.get("alignment_conc", 0.0))
    moving_groups: list[list[DAGOpNode]] = []
    moving_group_positions: list[list[tuple[int, int]]] = []
    moving_group_vectors: list[list[tuple[int, int]]] = []
    moving_group_qubit_ids: list[list[int]] = []
    used_mover_ids: set[int] = set()
    movement_time = 0 * (config["average_two_gate_time"] + config["t_switch"])

    # For each 2Q gate, attempt to fit it into an existing moving group based on AOD fit and movement vector compatibility.
    # If it doesn't fit in any existing group, start a new group with one of the atoms as the mover.
    # If multiple fit candidates exist for a group, prefer those that are parallel and similar in magnitude to existing group vectors.
    for node in layer_nodes:
        _, _, qubit_ids = op_node_signature(node)
        if len(qubit_ids) != 2:
            continue

        q0_id, q1_id = qubit_ids
        if q0_id not in qubit_map:
            raise ValueError(f"Qubit id {q0_id} from DAG not found in placed qubits.")
        if q1_id not in qubit_map:
            raise ValueError(f"Qubit id {q1_id} from DAG not found in placed qubits.")

        q0_pos = qubit_map[q0_id].grid_position()
        q1_pos = qubit_map[q1_id].grid_position()
        candidates = [
            (q0_id, q0_pos, _movement_vector(q0_pos, q1_pos)),
            (q1_id, q1_pos, _movement_vector(q1_pos, q0_pos)),
        ]

        placed = False
        for group_idx, positions in enumerate(moving_group_positions):
            fit_candidates: list[tuple[int, tuple[int, int], tuple[int, int], float]] = []
            for candidate_id, candidate_pos, candidate_vec in candidates:
                if candidate_id in used_mover_ids:
                    continue
                if candidate_id in moving_group_qubit_ids[group_idx]:
                    continue
                if _is_opposite_direction(
                    moving_group_vectors[group_idx],
                    candidate_vec,
                    alignment_conc,
                ):
                    # Candidate movement conflicts with this group's allowed alignment.
                    # Try a different group or force a new one.
                    continue
                if _fits_same_aod(positions, candidate_pos, max_row_span, max_col_span):
                    fit_candidates.append(
                        (
                            candidate_id,
                            candidate_pos,
                            candidate_vec,
                            _vector_alignment_score(moving_group_vectors[group_idx], candidate_vec),
                        )
                    )

            if not fit_candidates:
                continue

            # Prefer vectors that are parallel and similar in magnitude to the group vectors.
            best_id, best_pos, best_vec, _ = max(fit_candidates, key=lambda item: item[3])
            moving_groups[group_idx].append(node)
            positions.append(best_pos)
            moving_group_vectors[group_idx].append(best_vec)
            moving_group_qubit_ids[group_idx].append(best_id)
            used_mover_ids.add(best_id)
            placed = True
            break

        if not placed:
            # New group defaults to first atom as mover; no group vector exists to compare yet.
            if q0_id not in used_mover_ids:
                default_id = q0_id
                default_pos = q0_pos
                default_vec = _movement_vector(q0_pos, q1_pos)
            elif q1_id not in used_mover_ids:
                default_id = q1_id
                default_pos = q1_pos
                default_vec = _movement_vector(q1_pos, q0_pos)
            else:
                default_id = q0_id
                default_pos = q0_pos
                default_vec = _movement_vector(q0_pos, q1_pos)
            moving_groups.append([node])
            moving_group_positions.append([default_pos])
            moving_group_vectors.append([default_vec])
            moving_group_qubit_ids.append([default_id])
            used_mover_ids.add(default_id)
    # Reorder each group by movement-vector coherence.
    layer_event_start = len(event_log)
    for group_idx, group in enumerate(moving_groups):
        vector_order = _sort_group_by_alignment(moving_group_vectors[group_idx])
        if vector_order:
            ordered_group = [group[idx] for idx in vector_order]
            ordered_vectors = [moving_group_vectors[group_idx][idx] for idx in vector_order]
            ordered_ids = [moving_group_qubit_ids[group_idx][idx] for idx in vector_order]
            ordered_positions = [moving_group_positions[group_idx][idx] for idx in vector_order]
        else:
            ordered_group = group
            ordered_vectors = moving_group_vectors[group_idx]
            ordered_ids = moving_group_qubit_ids[group_idx]
            ordered_positions = moving_group_positions[group_idx]
        original_positions = ordered_positions[:]
        group_event_start = len(event_log)
        #now the vectors are order by alignment, we can then move in the highway and emit gates as we go.
        #starting from (0,0) we can reduce the vectors to successive differences, then we can emit moves for each vector in order.
        #must ensure each move is on the odd-odd highway.
        move = [0, 0]
        #now to load the AOD with the first move.
        reduced_vectors = _start_(ordered_vectors, move, ordered_ids, ordered_positions, event_log)
        #shuttle will add moves to the event_log and update the move vector as it goes.  
        _shuttle_(
            reduced_vectors,
            move,
            ordered_ids,
            ordered_positions,
            ordered_group,
            event_log,
        )
        final_position_map = {qid: ordered_positions[idx] for idx, qid in enumerate(ordered_ids)}
        unresolved = set(ordered_ids)
        for event in reversed(event_log):
            if event[0] == "move" and event[1] in unresolved:
                final_position_map[event[1]] = event[3]
                unresolved.remove(event[1])
                if not unresolved:
                    break

        final_positions = [final_position_map[qid] for qid in ordered_ids]
        home_positions = [qubit_map[qid].grid_position() for qid in ordered_ids]
        _return_(ordered_ids, final_positions, home_positions, event_log)

        reference_id = ordered_ids[0]
        reference_path: list[tuple[int, int]] = [original_positions[0]]
        reference_cursor = original_positions[0]
        for event in event_log[group_event_start:]:
            if event[0] != "move" or event[1] != reference_id:
                continue
            _, _, start_pos, end_pos = event
            # Track one continuous trajectory only; groups may contain duplicate ids.
            if start_pos != reference_cursor:
                continue
            if start_pos == end_pos:
                continue
            reference_path.append(end_pos)
            reference_cursor = end_pos
        movement_time += _time_trapezoid_(reference_path, config)

    layer_time = len(moving_groups) * (config["average_two_gate_time"] + config["t_switch"]) + movement_time
    layer_steps = len(event_log) - layer_event_start
    return layer_steps, layer_time

#--------------------------------
#Now for timing
#--------------------------------
def _time_trapezoid_(
    moves: list[tuple[int, int]],
    config:dict,
):
    """Compute movement time for axis-aligned segments using a trapezoidal/triangular profile."""
    v = config["max_velocity"]        # Pint Quantity
    a = config["max_acceleration"]    # Pint Quantity
    transfer_time = config["transfer_SLM_AOD"]
    grid_spacing = config["rydberg_radius"]

    total_time = 0 * (v / a).units    # initializes time quantity (seconds)

    for i in range(len(moves) - 1):
        x1, y1 = moves[i]
        x2, y2 = moves[i + 1]
        
        # Rounding Up
        # If the Neutral atom is moving from an AOD-steered tweezer to an SLM trap or vice versa
        # We approximate that has time to transfer_SLM_AOD + move from x1,y1 to x2,y2 using trapezoid.   
        q1_even = (x1 % 2 == 0) or (y1 % 2 == 0)
        q2_even = (x2 % 2 == 0) or (y2 % 2 == 0)
        if q1_even or q2_even:
            total_time += transfer_time
        dx = (x2 - x1)
        dy = (y2 - y1)
        if dx != 0 and dy != 0:
            raise ValueError(
                f"Invalid move segment {moves[i]} -> {moves[i + 1]}: each move must be along one axis only."
            )
        if dx == 0 and dy == 0:
            continue

        # convert grid movement to physical distance
        d = (abs(dx) + abs(dy)) * grid_spacing/2

        d_accel = v**2 / (2 * a)

        if d > 2 * d_accel:
            t = 2 * (v / a) + (d - 2 * d_accel) / v
        else:
            v_peak = (a * d)**(0.5)
            t = 2 * (v_peak / a)

        total_time += t

    return total_time
