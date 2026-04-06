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
    moves: list[tuple[int, int]],
    moving_ids: list[int],
    origin_positions: list[tuple[int, int]],
    event_log: list,
) -> list[tuple[int, int]]:
    """Apply initial vector reduction and choose a good first move."""

    reduced_vectors = _reduce_vectors_for_start(moving_vectors, (moves[0][0], moves[0][1]))
    if origin_positions[0][0] % 2 == 1 or origin_positions[0][1] % 2 == 1:
        return reduced_vectors
    
    def apply_move(step):
        dx, dy = step
        for i, (vx, vy) in enumerate(reduced_vectors):
            r, c = origin_positions[i]
            start = (r, c)
            end = (r + dx, c + dy)

            origin_positions[i] = end
            if i == 0:
                reduced_vectors[i] = (vx - dx, vy - dy)
            event_log.append(("move", moving_ids[i], start, end))

        moves.append((dx, dy))
        return reduced_vectors

    # -------------------------------------------------
    # 1. Single mover: direct greedy step toward target
    # -------------------------------------------------
    if len(reduced_vectors) == 1:
        dx, dy = reduced_vectors[0]

        if abs(dx) > abs(dy):
            step = (1 if dx > 0 else -1, 0)
        else:
            step = (0, 1 if dy > 0 else -1)

        return apply_move(step)

    # -------------------------------------------------
    # 2. Multi-mover: choose best shared step
    # -------------------------------------------------

    # candidate unit moves
    candidates = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    def score(step):
        """Score = number of vectors (excluding first) that become |v|=1,
        considering only up to and including the first non-(0,0) vector."""
        
        sx, sy = step
        new_vectors = [(vx - sx, vy - sy) for vx, vy in moving_vectors]

        count = 0
        seen_nonzero = False

        for i, (vx, vy) in enumerate(new_vectors):
            if (vx, vy) != (0, 0):
                if seen_nonzero:
                    break  # already processed first non-zero → stop
                seen_nonzero = True

            if i != 0 and (abs(vx) + abs(vy) == 1):
                count += 1

        return count

    # pick best step
    best_step = max(candidates, key=score)

    # -------------------------------------------------
    # 3. Even-even bias (your parity heuristic)
    # -------------------------------------------------
    r0, c0 = origin_positions[0]
    if (r0 % 2 == 0) and (c0 % 2 == 0):
        # slight bias toward axis-aligned reduction of largest vector
        vx, vy = reduced_vectors[0]
        if abs(vx) > abs(vy):
            preferred = (1 if vx > 0 else -1, 0)
        else:
            preferred = (0, 1 if vy > 0 else -1)

        # override if equally good
        if score(preferred) >= score(best_step):
            best_step = preferred

    return apply_move(best_step)


def parity_route_moves(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    Return a sequence of (dx, dy) moves that route from start → end
    respecting (even,odd)/(odd,even) parity constraints.
    """

    r1, c1 = start
    r2, c2 = end

    dx = r2 - r1
    dy = c2 - c1

    def is_even_odd(r, c):
        return (r % 2 == 0 and c % 2 == 1) or (r % 2 == 1 and c % 2 == 0)

    start_parity = is_even_odd(r1, c1)
    end_parity = is_even_odd(r2, c2)

    moves = []
    
    # -------------------------------------------------
    # Case 4: returning to even-even trap site
    # -------------------------------------------------
    if r2 % 2 == 0 and c2 % 2 == 0:
        if (r1 % 2 == 1 and c1 % 2 == 0):
            # handle x first
            step_y = dy - (1 if dy >= 0 else -1)
            moves.append((0, step_y))
            moves.append((dx, 0))
            moves.append((0, dy - step_y))  # final ±1
        else:
            # handle y first
            step_x = dx - (1 if dx >= 0 else -1)
            moves.append((dx - step_x, 0))
            moves.append((0, dy))
            moves.append((step_x, 0))  # final ±1

        return moves

    # -------------------------------------------------
    # Case 1: parity flips → always 2 moves
    # -------------------------------------------------
    if start_parity != end_parity:
        if (r1 % 2 == 0 and c1 % 2 == 1):  # (even, odd)
            moves.append((dx, 0))
            moves.append((0, dy))
        else:  # (odd, even)
            moves.append((0, dy))
            moves.append((dx, 0))
        return moves

    # -------------------------------------------------
    # Case 2: same parity
    # -------------------------------------------------

    # If purely axis-aligned → 1 move
    if (dx == 0 and c1 % 2 == 1) or (dy == 0 and r1 % 2 == 1):
        moves.append((dx, dy))
        return moves

    # -------------------------------------------------
    # Case 3: same parity/return to even-even, need 3 moves
    # -------------------------------------------------

    # Decide which axis to split first
    if (r1 % 2 == 1 and c1 % 2 == 0):
        # handle x first
        step_y = dy - (1 if dy >= 0 else -1)
        moves.append((0, step_y))
        moves.append((dx, 0))
        moves.append((0, dy - step_y))  # final ±1
    else:
        # handle y first
        step_x = dx - (1 if dx >= 0 else -1)
        moves.append((dx - step_x, 0))
        moves.append((0, dy))
        moves.append((step_x, 0))  # final ±1

    return moves

def _shuttle_(
    moving_vectors: list[tuple[int, int]],
    moves: list[tuple[int, int]],
    moving_ids: list[int],
    moving_group_positions: list[tuple[int, int]],
    gate_nodes: list,
    event_log: list,
) -> None:
    """Route all movers toward targets while prioritizing parallel-ready states."""

    # -------------------------------------------------
    # Score: stop after FIRST non-(0,0), inclusive
    # -------------------------------------------------
    def score(step, idx):
        """Score = number of vectors (excluding idx) that become |v|=1,
        starting from idx and stopping after first non-(0,0) (inclusive)."""
        
        sx, sy = step
        new_vectors = [(vx - sx, vy - sy) for vx, vy in moving_vectors]

        count = 0
        seen_nonzero = False

        for i in range(idx, len(new_vectors)):
            vx, vy = new_vectors[i]

            if (vx, vy) != (0, 0):
                if seen_nonzero:
                    break  # already processed first non-zero → stop
                seen_nonzero = True

            if i != idx and (abs(vx) + abs(vy) == 1):
                count += 1

        return count

    # -------------------------------------------------
    # Greedy step (your simplified routing logic)
    # -------------------------------------------------
    def greedy_step(idx):
        """
        Choose the best unit step for mover `idx` by minimizing
        remaining parity-route length.
        """

        q1_pos = moving_group_positions[idx]
        vx, vy = moving_vectors[idx]
        q2_pos = (q1_pos[0] + vx, q1_pos[1] + vy)

        candidates = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        best_step = None
        best_cost = float("inf")

        for step in candidates:
            sx, sy = step

            # simulate move
            new_pos = (q2_pos[0] + sx, q2_pos[1] + sy)

            # compute remaining path length
            route = parity_route_moves(q1_pos, new_pos)
            cost = len(route)

            if cost < best_cost:
                best_cost = cost
                best_step = step

        return best_step

    # -------------------------------------------------
    # Main loop
    # -------------------------------------------------
    for idx, (vx, vy) in enumerate(moving_vectors):
        if (vx, vy) == (0,0):
            if idx < len(gate_nodes):
                gate_name, gate_params, qubit_ids = op_node_signature(gate_nodes[idx])
                gate_line = format_gate_line(gate_name, gate_params, qubit_ids)
                if event_log and event_log[-1][0] == "gate":
                    event_log[-1] = ("gate", f"{event_log[-1][1]} {gate_line}")
                else:
                    event_log.append(("gate", gate_line))
            continue
        if (vx, vy) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if idx < len(gate_nodes):
                gate_name, gate_params, qubit_ids = op_node_signature(gate_nodes[idx])
                gate_line = format_gate_line(gate_name, gate_params, qubit_ids)
                if event_log and event_log[-1][0] == "gate":
                    event_log[-1] = ("gate", f"{event_log[-1][1]} {gate_line}")
                else:
                    event_log.append(("gate", gate_line))
            if idx + 1 < len(moving_vectors):
                next_vx, next_vy = moving_vectors[idx + 1]
                moving_vectors[idx + 1] = (next_vx + vx, next_vy + vy)
            continue

        candidates = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        q1_pos = moving_group_positions[idx]
        q2_pos = (q1_pos[0] + vx, q1_pos[1] + vy)

        best_step = max(candidates, key=lambda step: score(step, idx))
        best_score = score(best_step, idx)
        # fallback if no parallel benefit
        if best_score == 0:
            best_step = greedy_step(idx)
        
        new_pos = (q2_pos[0] + best_step[0], q2_pos[1] + best_step[1])
        if idx + 1 < len(moving_vectors):
                next_vx, next_vy = moving_vectors[idx + 1]
                moving_vectors[idx + 1] = (next_vx - best_step[0], next_vy - best_step[1])
        next_move = parity_route_moves(q1_pos, new_pos)
        for step in next_move:
            sx, sy = step
            for atom_i, atom_id in enumerate(moving_ids):
                start = moving_group_positions[atom_i]
                end = (start[0] + sx, start[1] + sy)
                event_log.append(("move", atom_id, start, end))
                moving_group_positions[atom_i] = end
            moves.append(step)

        if idx < len(gate_nodes):
            gate_name, gate_params, qubit_ids = op_node_signature(gate_nodes[idx])
            gate_line = format_gate_line(gate_name, gate_params, qubit_ids)
            event_log.append(("gate", gate_line))


def _return_(
    moving_ids: list[int],
    moves: list[tuple[int, int]],
    current_positions: list[tuple[int, int]],
    home_positions: list[tuple[int, int]],
    event_log: list[ScheduleEvent],
) -> None:
    """Log return moves for all moving qubits, synchronized when possible."""
    if not moving_ids:
        return
    if len(moving_ids) != len(current_positions) or len(moving_ids) != len(home_positions):
        raise ValueError("Return path inputs must have matching lengths.")

    current_position = current_positions[0]
    home_position = home_positions[0]
    next_move = parity_route_moves(current_position, home_position)
    for step in next_move:
        sx, sy = step
        for atom_i, atom_id in enumerate(moving_ids):
            start = current_positions[atom_i]
            end = (start[0] + sx, start[1] + sy)
            event_log.append(("move", atom_id, start, end))
            current_positions[atom_i] = end
        moves.append(step)



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
    Previous_Ids: list[int] | None = None,
    Previous_Positions: list[tuple[int, int]] | None = None,
    current_positions: dict[int, tuple[int, int]] | None = None,
) -> tuple[int, Any, list[tuple[int, int]], list[int]]:
    """Group 2Q gates by AOD fit and append grouped gate steps into ``event_log`` in place.

    One atom per gate is selected as the moving candidate based on AOD fit and movement
    vector compatibility with existing atoms already assigned to a moving group.
    Returns the number of emitted gate timesteps and time contribution.
    """
    if not layer_nodes:
        zero_time = 0 * (config["average_two_gate_time"] + config["t_switch"])
        return 0, zero_time, [], []

    qubit_map = {q.id: q for q in qubits}
    if current_positions is None:
        current_positions = {q.id: q.grid_position() for q in qubits}
    max_row_span, max_col_span = _max_grid_spans(config["max_dimension"])
    alignment_conc = float(config.get("alignment_conc", 0.0))
    allow_parallel = bool(config.get("parallel", False))
    moving_groups: list[list[DAGOpNode]] = []
    moving_group_positions: list[list[tuple[int, int]]] = []
    moving_group_vectors: list[list[tuple[int, int]]] = []
    moving_group_qubit_ids: list[list[int]] = []
    used_mover_ids: set[int] = set()
    movement_time = 0 * (config["average_two_gate_time"] + config["t_switch"])
    last_group_ordered_positions: list[tuple[int, int]] = []
    last_group_ordered_ids: list[int] = []
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

        q0_pos = current_positions.get(q0_id, qubit_map[q0_id].grid_position())
        q1_pos = current_positions.get(q1_id, qubit_map[q1_id].grid_position())
        if not allow_parallel:
            # Sequential mode: one mover per gate group.
            if (q0_pos[0] % 2 != 0) or (q0_pos[1] % 2 != 0):
                default_id = q0_id
                default_pos = q0_pos
                default_vec = _movement_vector(q0_pos, q1_pos)
            elif (q1_pos[0] % 2 != 0) or (q1_pos[1] % 2 != 0):
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
            continue

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
    # If previous mover ids appear as a full group again, process that group first.
    if Previous_Ids:
        prev_set = set(Previous_Ids)
        front_idx = None
        for gi, group_ids in enumerate(moving_group_qubit_ids):
            if set(group_ids) == prev_set:
                front_idx = gi
                break
        if front_idx is not None and front_idx != 0:
            moving_groups.insert(0, moving_groups.pop(front_idx))
            moving_group_positions.insert(0, moving_group_positions.pop(front_idx))
            moving_group_vectors.insert(0, moving_group_vectors.pop(front_idx))
            moving_group_qubit_ids.insert(0, moving_group_qubit_ids.pop(front_idx))
    prev_set = set(Previous_Ids) if Previous_Ids else set()

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
        #--------------------------
        #now the vectors are order by alignment, we can then move in the highway and emit gates as we go.
        #--------------------------
        #starting from (0,0) we can reduce the vectors to successive differences, then we can emit moves for each vector in order.
        #must ensure each move is on the odd-odd highway.

        moves = [[0, 0]] #thus is mostly for timing.
        #now to load the AOD with the first move.
        same_as_previous = bool(prev_set) and set(ordered_ids) == prev_set
        if not same_as_previous:
            if Previous_Ids and Previous_Positions and len(Previous_Ids) == len(Previous_Positions):
                prev_ids = [qid for qid in Previous_Ids if qid in qubit_map]
                prev_current_positions = [Previous_Positions[i] for i, qid in enumerate(Previous_Ids) if qid in qubit_map]
                prev_home_positions = [qubit_map[qid].grid_position() for qid in prev_ids]
                if prev_ids:
                    _return_(prev_ids, moves, prev_current_positions, prev_home_positions, event_log)
        else:
            prev_pos_by_id = {Previous_Ids[i]: Previous_Positions[i] for i in range(len(Previous_Ids))}
            prev_home_by_id = {qid: qubit_map[qid].grid_position() for qid in Previous_Ids if qid in qubit_map}
            if ordered_ids and ordered_vectors:
                first_id = ordered_ids[0]
                if first_id in prev_pos_by_id and first_id in prev_home_by_id:
                    prev_r, prev_c = prev_pos_by_id[first_id]
                    home_r, home_c = prev_home_by_id[first_id]
                    offset = (prev_r - home_r, prev_c - home_c)
                    ordered_vectors = [
                        (vx - offset[0], vy - offset[1]) for vx, vy in ordered_vectors
                    ]
            ordered_positions = [prev_pos_by_id.get(qid, ordered_positions[i]) for i, qid in enumerate(ordered_ids)]

        #------------
        #Returned previous group to original positions.
        #Now to move the current group
        #------------

        reduced_vectors = _start_(ordered_vectors, moves, ordered_ids, ordered_positions, event_log)
        #shuttle will add moves to the event_log and update the moves.  
        _shuttle_(
            reduced_vectors,
            moves,
            ordered_ids,
            ordered_positions,
            ordered_group,
            event_log,
        )
        movement_time += _time_trapezoid_(moves, config, add_transfer_time= (not same_as_previous))

        Previous_Positions = ordered_positions[:]
        Previous_Ids = ordered_ids[:]



    layer_time = len(moving_groups) * (config["average_two_gate_time"] + config["t_switch"]) + movement_time
    layer_steps = len(event_log) - layer_event_start
    return layer_steps, layer_time, Previous_Positions, Previous_Ids

#--------------------------------
#Now for timing
#--------------------------------
def _time_trapezoid_(
    moves: list[tuple[int, int]],
    config:dict,
    add_transfer_time: bool = True,
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
        if add_transfer_time:
            total_time += transfer_time
        dx = (x2 - x1)
        dy = (y2 - y1)
    
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
