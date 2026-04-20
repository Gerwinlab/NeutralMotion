from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
from math import exp, hypot, log
import random
from typing import Iterable, Mapping

from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import extract_index_from_bit
from .grid import Qubit, naive_fill


@dataclass(frozen=True)
class LayerVectorStats:
    """Derived vector metrics for a single 2Q layer."""

    vectors: list[tuple[int, int]]
    vector_alignment_score: float
    longest_magnitude: float
    max_parallel_gates: int
    layer_cost: float


@dataclass(frozen=True)
class FastSAResult:
    """Result bundle for full 3-stage Fast-SA."""
    current_cost: float
    best_cost: float
    current_positions: dict[int, tuple[int, int]]
    best_positions: dict[int, tuple[int, int]]
    stage1_iterations: int
    stage2_iterations: int
    stage3_iterations: int



def qubit_position_map(qubits: Iterable[Qubit]) -> dict[int, tuple[int, int]]:
    """Map logical qubit id to integer grid coordinate (x, y) = (col, row)."""
    pos: dict[int, tuple[int, int]] = {}
    for qubit in qubits:
        row, col = qubit.grid_position()
        pos[qubit.id] = (col, row)
    return pos


def position_map_from_grid(grid) -> dict[int, tuple[int, int]]:
    """Build qubit position map directly from a populated grid."""
    positions: dict[int, tuple[int, int]] = {}
    for row in grid:
        for node in row:
            if getattr(node, "qubit", None) is None:
                continue
            qubit_obj = node.qubit
            qubit_id = getattr(qubit_obj, "id", None)
            if qubit_id is None:
                # Fallback for int-typed qubit slots.
                qubit_id = int(qubit_obj)
            positions[int(qubit_id)] = (node.col, node.row)
    return positions



def vector_alignment_score(vectors: Iterable[tuple[int, int]]) -> float:
    """Alignment score = sum of unit-direction dot-products across all pairs."""
    v = list(vectors)
    if len(v) < 2:
        return 0.0
    normalizer = 0.0
    score = 0.0
    for i in range(len(v)):
        ux, uy = v[i]
        for j in range(i + 1, len(v)):
            vx, vy = v[j]
            umag = hypot(ux, uy)
            vmag = hypot(vx, vy)
            cosine = abs((ux * vx + uy * vy) / (umag * vmag))
            mag_ratio = min(umag, vmag) / max(umag, vmag)
            normalizer +=1
            score += cosine * mag_ratio
            #In the ideal case that all vectors are identical, then we have cosine=1 and mag_ratio=1 for all pairs, so the score grows quadratically with the number of vectors. In a less ideal case where we have some parallel but differently sized vectors, the mag_ratio will reduce the contribution of misaligned pairs, while still rewarding parallelism among similarly sized vectors.
    return score/normalizer #must always be between 0 and 1.

#--------------------------
#Computing costs
#--------------------------
def compute_fastsa_cost(
    two_qubit_layers: Iterable[Iterable["DAGOpNode"]],
    positions_or_qubits: Mapping[int, tuple[int, int]] | Iterable["Qubit"],normalize_by_length : int = 1
) -> tuple[float, list[dict]]:
    """Compute Fast-SA cost"""
    if isinstance(positions_or_qubits, Mapping):
        positions = dict(positions_or_qubits)
    else:
        positions = qubit_position_map(positions_or_qubits)

    def canonical(v):
        x, y = v
        if x < 0 or (x == 0 and y < 0):
            return (-x, -y)
        return (x, y)


    total_cost = 0.0
    normalize_by_layers = len(list(two_qubit_layers.layers()))
    layer_stats_list: list[dict] = []
    for layer_nodes in two_qubit_layers.layers():
        vectors: list[tuple[int,int]] = []
        layer_cost = 0.0
        for node in layer_nodes["graph"].op_nodes():
            if len(node.qargs) != 2:
                continue
            q0 = extract_index_from_bit(node.qargs[0])
            q1 = extract_index_from_bit(node.qargs[1])
            if q0 not in positions or q1 not in positions:
                raise KeyError(f"Missing qubit positions for q[{q0}] or q[{q1}].")
            
            x0, y0 = positions[q0]
            x1, y1 = positions[q1]
            vectors.append((x1 - x0, y1 - y0))
        # --- Finished making the vectors ---
        alignment = vector_alignment_score(vectors)
        #length of vector normalized by the multiplicity
        
        counts = Counter(canonical(v) for v in vectors)
        norm_counts = len(counts)
        for vec, count in counts.items():
            mag = hypot(vec[0], vec[1])
            layer_cost += mag * (1.0 / count) / normalize_by_length /norm_counts #ensure below 1
        total_cost += (layer_cost - alignment)/normalize_by_layers
        layer_stats_list.append({
            "vectors": vectors,
            "vector_alignment_score": alignment,
            "layer_cost": layer_cost,
        })
    return total_cost, layer_stats_list
#--------------------------

#--------------------------


def _sample_random_positions_with_naive_fill(
    grid,
    num_qubits: int,
    seed: int,
) -> dict[int, tuple[int, int]]:
    """Sample one random placement by running naive_fill on a fresh grid copy."""
    grid_copy = deepcopy(grid)
    qubits = naive_fill(grid_copy, num_qubits, seed=seed, random_fill=True)
    return qubit_position_map(qubits)


def _temperature_for_step(
    n: int,
    *,
    average_uphill_cost: float,
    initial_accept_prob: float,
    average_cost_change: float,
    k: int,
    c: float,
) -> float:
    denom = log(initial_accept_prob)

    # ensure denom is not zero
    if abs(denom) < 1e-12:
        denom = -1e-12

    # make temperature positive
    t1 = -average_uphill_cost / denom

    if n <= 1:
        return t1
    if n <= k:
        return max((t1 * average_cost_change) / (n * c), 1e-12)
    return max((t1 * average_cost_change) / n, 1e-12)


def _acceptance_probability(delta_c: float, temperature: float) -> float:
    """Stable Metropolis acceptance probability."""
    if delta_c <= 0:
        return 1.0
    if temperature <= 0:
        return abs(temperature)
    exponent = -delta_c / temperature
    if exponent < -700:
        return 0.0
    return min(1.0, exp(exponent))


def greedy_swap_neighbor(
    current_positions: dict[int, tuple[int, int]],
    two_qubit_dag: DAGCircuit,
) -> tuple[dict[int, tuple[int, int]], float]:
    """Greedy neighbor: swap two active qubits from the worst-cost layer."""

    current_cost, layer_stats = compute_fastsa_cost(two_qubit_dag, current_positions)
    layers = [layer["graph"].op_nodes() for layer in two_qubit_dag.layers()]

    if not layer_stats or not layers:
        return dict(current_positions), current_cost

    # Find worst layer
    worst_layer_idx = max(
        range(len(layer_stats)),
        key=lambda idx: layer_stats[idx]["layer_cost"],
    )
    worst_layer_nodes = layers[worst_layer_idx]

    # Collect active qubits
    active_qids: list[int] = []
    for node in worst_layer_nodes:
        if len(node.qargs) != 2:
            continue
        q0 = extract_index_from_bit(node.qargs[0])
        q1 = extract_index_from_bit(node.qargs[1])

        if q0 not in active_qids:
            active_qids.append(q0)
        if q1 not in active_qids:
            active_qids.append(q1)

    if len(active_qids) < 2:
        return dict(current_positions), current_cost

    best_positions = dict(current_positions)
    best_cost = float('inf')

    # Try all swaps (inline swap logic)
    for i in range(len(active_qids)):
        for j in range(i + 1, len(active_qids)):
            qa = active_qids[i]
            qb = active_qids[j]

            candidate = dict(current_positions)
            candidate[qa], candidate[qb] = candidate[qb], candidate[qa]

            candidate_cost, _ = compute_fastsa_cost(two_qubit_dag, candidate)

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_positions = candidate

    return best_positions, best_cost


def greedy_max_parallel(
    random_layer: Iterable["DAGOpNode"],
    current_positions: Mapping[int, tuple[int, int]],
    *,
    grid_shape: tuple[int, int],
) -> dict[int, tuple[int, int]]:
    """
    For one layer, compute each gate vector from current q0/q1 positions, pick
    the most common vector, and adjust mismatched gates by moving either q0 or
    q1 to realize that vector.

    Coordinates are (col, row). A move is accepted only if the new position is:
    1) inside the grid, 2) not already occupied by another qubit, and

    """
    positions = dict(current_positions)
    grid_rows, grid_cols = grid_shape


    gate_pairs: list[tuple[int, int, tuple[int, int]]] = []
    for node in random_layer:
        if len(node.qargs) != 2:
            continue
        q0 = extract_index_from_bit(node.qargs[0])
        q1 = extract_index_from_bit(node.qargs[1])
        if q0 not in positions or q1 not in positions:
            continue
        x0, y0 = positions[q0]
        x1, y1 = positions[q1]
        gate_pairs.append((q0, q1, (x1 - x0, y1 - y0)))

    if not gate_pairs:
        return positions

    vector_counts = Counter(vec for _, _, vec in gate_pairs)
    target_vec, _ = max(
        vector_counts.items(),
        key=lambda item: (item[1], -hypot(item[0][0], item[0][1])),
    )
    target_dx, target_dy = target_vec

    # Track occupancy as we update placements gate-by-gate.
    occupied: dict[tuple[int, int], int] = {xy: qid for qid, xy in positions.items()}

    def _in_bounds(x: int, y: int) -> bool:
        return 0 <= x < grid_cols and 0 <= y < grid_rows


    def _move_or_swap(qid: int, target: tuple[int, int]) -> None:
        """Move qid to target; if occupied, swap the two qubits' positions."""
        old = positions[qid]
        other = occupied.get(target)
        if other is None or other == qid:
            if old in occupied and occupied[old] == qid:
                del occupied[old]
            positions[qid] = target
            occupied[target] = qid
            return

        positions[qid], positions[other] = positions[other], positions[qid]
        occupied[positions[qid]] = qid
        occupied[positions[other]] = other

    for q0, q1, vec in gate_pairs:
        if vec == target_vec:
            continue

        x0, y0 = positions[q0]
        x1, y1 = positions[q1]

        # Option 1: keep q0 fixed and move q1 (swap if destination occupied).
        cand_q1 = (x0 + target_dx, y0 + target_dy)
        can_move_q1 = _in_bounds(cand_q1[0], cand_q1[1])

        if can_move_q1:
            _move_or_swap(q1, cand_q1)
            continue

        # Option 2: keep q1 fixed and move q0 (swap if destination occupied).
        cand_q0 = (x1 - target_dx, y1 - target_dy)
        can_move_q0 = _in_bounds(cand_q0[0], cand_q0[1])

        if can_move_q0:
            _move_or_swap(q0, cand_q0)
    
    return positions


def _randomize_small_section(
    positions: dict[int, tuple[int, int]],
    *,
    rng: random.Random,
    section_size: int = 4,
) -> dict[int, tuple[int, int]]:
    """Randomly permute positions among a small subset of qubits."""
    qids = list(positions.keys())
    if len(qids) <= 1:
        return dict(positions)
    m = min(max(section_size, 2), len(qids))
    subset = rng.sample(qids, m)
    shuffled_positions = [positions[qid] for qid in subset]
    rng.shuffle(shuffled_positions)
    randomized = dict(positions)
    for idx, qid in enumerate(subset):
        randomized[qid] = shuffled_positions[idx]
    return randomized


def run_fastsa(
    grid,
    num_qubits: int,
    two_qubit_dag: DAGCircuit,
    config: Mapping[str, object],
    *,
    stage1_iterations: int = 10,
    stage2_iterations: int = 100,
    initial_accept_prob: float = 0.99,
    c: float = 100.0,
    stage3_temperature_threshold: float = 1e-3,
    stage3_max_iterations: int = 1000,
    stage3_section_size: int = 4,
    seed: int = 0,
) -> FastSAResult:
    """Run 3-stage Fast-SA: random stage, greedy swap stage, local-random stage."""
    if stage1_iterations < 1:
        raise ValueError("stage1_iterations must be >= 1")
    if stage2_iterations < 0:
        raise ValueError("stage2_iterations must be >= 0")
    if stage3_max_iterations < 0:
        raise ValueError("stage3_max_iterations must be >= 0")
    if not (0.0 < initial_accept_prob < 1.0):
        raise ValueError("initial_accept_prob must be in (0, 1)")
    if c <= 0:
        raise ValueError("c must be > 0, you want c to be large to reduce uphill acceptance in the second stage")

    rng = random.Random(seed)
    #The Grid comes in empty, so do not want to copy to the original until the end.

    current_positions = _sample_random_positions_with_naive_fill(
        grid,
        num_qubits,
        seed=rng.randint(0, 2**31 - 1), #pick a random seed every time we call this.
    )

    current_cost, _ = compute_fastsa_cost(two_qubit_dag, current_positions)
    best_positions = dict(current_positions)
    best_cost = current_cost

    n = 1

    uphill_deltas: list[float] = []
    abs_delta_samples: list[float] = []
    avg_uphill = 1.0
    avg_delta = 1.0
    temperature = _temperature_for_step(
                n,
                average_uphill_cost=avg_uphill,
                initial_accept_prob=initial_accept_prob,
                average_cost_change=avg_delta,
                k=stage2_iterations+stage1_iterations,
                c=c,
            )
    
    def _record_delta(delta_c: float, *, accepted: bool) -> float:
        nonlocal avg_uphill, avg_delta, current_cost, current_positions, best_cost, best_positions
        temperature =0.0
        if delta_c > 0:
            uphill_deltas.append(delta_c)
            avg_uphill = sum(uphill_deltas) / len(uphill_deltas)
        if accepted:
            abs_delta_samples.append(delta_c)
            avg_delta = sum(abs(delta_c) for delta_c in abs_delta_samples) / len(abs_delta_samples)
        temperature = _temperature_for_step(
            n,
            average_uphill_cost=avg_uphill,
            initial_accept_prob=initial_accept_prob,
            average_cost_change=avg_delta,
            k=stage2_iterations+stage1_iterations,
            c=c,
        )
        return temperature

    # Stage 1: n=1 initialization regime with random neighboring placements.
    for step in range(stage1_iterations):
        # sample a completely new layout
        neighbor_positions = _sample_random_positions_with_naive_fill(
            grid,
            num_qubits,
            seed=rng.randint(0, 2**31 - 1),
        )

        neighbor_cost, _ = compute_fastsa_cost(two_qubit_dag, neighbor_positions)

        delta_c = neighbor_cost - current_cost
        accept_prob = _acceptance_probability(delta_c, temperature)

        accepted = rng.random() < accept_prob

        if accepted:
            current_positions = neighbor_positions
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(current_positions)
        n+=1
        temperature = _record_delta(delta_c, accepted=accepted)

    # Stage 2: greedy swap search for k iterations.
    stage2_layers = [layer["graph"].op_nodes() for layer in two_qubit_dag.layers()]
    dims = config["dimensions"]
    grid_shape = (2 * int(dims[0]) - 1, 2 * int(dims[1]) - 1)
    for _ in range(stage2_iterations):
        # for one iteration of the second stage
        #first swap within the worst layer. 
        neighbor_positions, neighbor_cost = greedy_swap_neighbor(current_positions, two_qubit_dag)
        delta_c = neighbor_cost - current_cost
        accept_prob = _acceptance_probability(delta_c, temperature)
        accepted = rng.random() < accept_prob
        if accepted:
            current_positions = neighbor_positions
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(current_positions)
        
        if stage2_layers:
            random_layer = rng.choice(stage2_layers)
            neighbor_positions = greedy_max_parallel(
                random_layer,
                current_positions,
                grid_shape=grid_shape
            )

            neighbor_cost, _ = compute_fastsa_cost(two_qubit_dag, neighbor_positions)
            delta_c = neighbor_cost - current_cost
            accept_prob = _acceptance_probability(delta_c, temperature)

            accepted = rng.random() < accept_prob
            if accepted:
                current_positions = neighbor_positions
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_positions = dict(current_positions)
        n+=1
        temperature = _record_delta(delta_c, accepted=accepted)

    # Stage 3: local randomization until temperature drops below threshold.
    stage3_iterations = 0
    while stage3_iterations < stage3_max_iterations:
        if abs(temperature) < stage3_temperature_threshold:
            break
        neighbor_positions = _randomize_small_section(
            current_positions,
            rng=rng,
            section_size=stage3_section_size,
        )
        neighbor_cost, _ = compute_fastsa_cost(two_qubit_dag, neighbor_positions)
        delta_c = neighbor_cost - current_cost
        accept_prob = _acceptance_probability(delta_c, temperature)
        accepted = rng.random() < accept_prob
        if accepted:
            current_positions = neighbor_positions
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(current_positions)
        temperature = _record_delta(delta_c, accepted=accepted)
        n+=1
        stage3_iterations += 1

    return FastSAResult(
        current_cost=current_cost,
        best_cost=best_cost,
        current_positions=current_positions,
        best_positions=best_positions,
        stage1_iterations=stage1_iterations,
        stage2_iterations=stage2_iterations,
        stage3_iterations=stage3_iterations
    )
