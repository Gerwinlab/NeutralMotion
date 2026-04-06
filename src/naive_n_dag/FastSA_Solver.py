from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
from math import exp, hypot, log
import random
from typing import Iterable

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
class FastSAStage1Result:
    """Result bundle for first-stage random-placement Fast-SA."""

    accepted_n: int
    iterations: int
    acceptance_count: int
    current_cost: float
    best_cost: float
    current_positions: dict[int, tuple[int, int]]
    best_positions: dict[int, tuple[int, int]]
    temperature_history: list[float]
    delta_history: list[float]


def random_initial_placement(
    grid,
    num_qubits: int,
    *,
    seed: int = 0,
) -> list[Qubit]:
    """Generate a random initial placement using ``naive_fill``."""
    return naive_fill(grid, num_qubits, seed=seed, random_fill=True)


def qubit_position_map(qubits: Iterable[Qubit]) -> dict[int, tuple[int, int]]:
    """Map logical qubit id to integer grid coordinate (x, y) = (col, row)."""
    pos: dict[int, tuple[int, int]] = {}
    for qubit in qubits:
        row, col = qubit.grid_position()
        pos[qubit.id] = (col, row)
    return pos


def gate_vector_from_node(
    node: DAGOpNode,
    positions: dict[int, tuple[int, int]],
) -> tuple[int, int]:
    """Return the 2Q gate vector from first qarg to second qarg."""
    if len(node.qargs) != 2:
        raise ValueError("Expected a 2-qubit node for vector computation.")
    q0 = extract_index_from_bit(node.qargs[0])
    q1 = extract_index_from_bit(node.qargs[1])
    if q0 not in positions or q1 not in positions:
        raise KeyError(f"Missing qubit positions for q[{q0}] or q[{q1}].")
    x0, y0 = positions[q0]
    x1, y1 = positions[q1]
    return (x1 - x0, y1 - y0)


def layer_gate_vectors(
    layer_nodes: Iterable[DAGOpNode],
    positions: dict[int, tuple[int, int]],
) -> list[tuple[int, int]]:
    """Compute all 2Q vectors for a layer."""
    vectors: list[tuple[int, int]] = []
    for node in layer_nodes:
        if len(node.qargs) != 2:
            continue
        vectors.append(gate_vector_from_node(node, positions))
    return vectors


def vector_alignment_score(vectors: Iterable[tuple[int, int]]) -> float:
    """Alignment score = sum of unit-direction dot-products across all pairs."""
    v = list(vectors)
    if len(v) < 2:
        return 0.0

    unit_dirs: list[tuple[float, float]] = []
    for dx, dy in v:
        mag = hypot(dx, dy)
        if mag == 0:
            unit_dirs.append((0.0, 0.0))
        else:
            unit_dirs.append((dx / mag, dy / mag))

    score = 0.0
    for i in range(len(unit_dirs)):
        ux, uy = unit_dirs[i]
        for j in range(i + 1, len(unit_dirs)):
            vx, vy = unit_dirs[j]
            score += ux * vx + uy * vy
    return score


def _count_axis_offset_two(vectors: Iterable[tuple[int, int]]) -> int:
    """Count vectors with +/-2 on either axis."""
    return sum(1 for dx, dy in vectors if abs(dx) == 2 or abs(dy) == 2)


def max_parallel_gates(vectors: Iterable[tuple[int, int]]) -> int:
    """Compute parallel-gate count as: max-identical + max(+/-2-axis subset)."""
    v = list(vectors)
    if not v:
        return 0
    identical_max = max(Counter(v).values())
    offset_two_max = _count_axis_offset_two(v)
    return identical_max + offset_two_max


def longest_gate_vector_magnitude(vectors: Iterable[tuple[int, int]]) -> float:
    """Return the largest Euclidean norm among gate vectors."""
    longest = 0.0
    for dx, dy in vectors:
        longest = max(longest, hypot(dx, dy))
    return longest


def layer_vector_stats(
    layer_nodes: Iterable[DAGOpNode],
    positions: dict[int, tuple[int, int]],
) -> LayerVectorStats:
    """Compute layer vector metrics and the per-layer cost term."""
    vectors = layer_gate_vectors(layer_nodes, positions)
    alignment = vector_alignment_score(vectors)
    longest = longest_gate_vector_magnitude(vectors)
    parallel = max_parallel_gates(vectors)

    if parallel <= 0:
        layer_cost = 0.0
    else:
        layer_cost = (longest / parallel) - alignment

    return LayerVectorStats(
        vectors=vectors,
        vector_alignment_score=alignment,
        longest_magnitude=longest,
        max_parallel_gates=parallel,
        layer_cost=layer_cost,
    )


def fastsa_cost_for_layers(
    two_qubit_layers: Iterable[Iterable[DAGOpNode]],
    positions: dict[int, tuple[int, int]],
) -> tuple[float, list[LayerVectorStats]]:
    """Sum layer costs across provided 2Q layers."""
    total = 0.0
    stats: list[LayerVectorStats] = []
    for layer_nodes in two_qubit_layers:
        layer_stats = layer_vector_stats(layer_nodes, positions)
        stats.append(layer_stats)
        total += layer_stats.layer_cost
    return total, stats


def fastsa_cost_for_two_qubit_dag(
    two_qubit_dag: DAGCircuit,
    positions: dict[int, tuple[int, int]],
) -> tuple[float, list[LayerVectorStats]]:
    """Sum Fast-SA cost over all layers in a Qiskit two-qubit DAG."""
    layers = [layer["graph"].op_nodes() for layer in two_qubit_dag.layers()]
    return fastsa_cost_for_layers(layers, positions)


def _placement_signature(positions: dict[int, tuple[int, int]]) -> tuple[tuple[int, int, int], ...]:
    return tuple((qid, xy[0], xy[1]) for qid, xy in sorted(positions.items()))


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
    t1: float,
    average_cost_change: float,
    k: int,
    c: float,
) -> float:
    if n <= 1:
        return max(t1, 1e-12)
    if n <= k:
        return max((t1 * average_cost_change) / (n * c), 1e-12)
    return max((t1 * average_cost_change) / n, 1e-12)


def run_fastsa_stage1_random_placements(
    grid,
    num_qubits: int,
    two_qubit_dag: DAGCircuit,
    *,
    iterations: int = 200,
    initial_accept_prob: float = 0.99,
    k: int = 100,
    c: float = 100.0,
    seed: int = 0,
) -> FastSAStage1Result:
    """Run first-stage Fast-SA using random placements as neighboring states.

    Acceptance rule:
    - Let ``deltaC = C_neighbor - C_current``.
    - Accept with probability ``min(1, exp(-deltaC / T_n))``.
    - Increment ``n`` only when a placement is accepted.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    if not (0.0 < initial_accept_prob < 1.0):
        raise ValueError("initial_accept_prob must be in (0, 1)")
    if k < 1:
        raise ValueError("k must be >= 1")
    if c <= 0:
        raise ValueError("c must be > 0")

    rng = random.Random(seed)

    current_positions = _sample_random_positions_with_naive_fill(
        grid,
        num_qubits,
        seed=rng.randint(0, 2**31 - 1),
    )
    current_cost, _ = fastsa_cost_for_two_qubit_dag(two_qubit_dag, current_positions)
    best_positions = dict(current_positions)
    best_cost = current_cost

    accepted_n = 1
    acceptance_count = 0
    temperature_history: list[float] = []
    delta_history: list[float] = []

    uphill_deltas: list[float] = []
    abs_delta_samples: list[float] = []
    avg_uphill = 1.0
    avg_abs_delta = 1.0
    t1 = avg_uphill / abs(log(initial_accept_prob))

    for _ in range(iterations):
        # Ensure the "two different random placements" condition for each proposal.
        neighbor_positions = current_positions
        for _attempt in range(10):
            candidate_positions = _sample_random_positions_with_naive_fill(
                grid,
                num_qubits,
                seed=rng.randint(0, 2**31 - 1),
            )
            if _placement_signature(candidate_positions) != _placement_signature(current_positions):
                neighbor_positions = candidate_positions
                break

        neighbor_cost, _ = fastsa_cost_for_two_qubit_dag(two_qubit_dag, neighbor_positions)
        delta_c = neighbor_cost - current_cost
        delta_history.append(delta_c)

        abs_delta_samples.append(abs(delta_c))
        avg_abs_delta = sum(abs_delta_samples) / len(abs_delta_samples)
        if delta_c > 0:
            uphill_deltas.append(delta_c)
            avg_uphill = sum(uphill_deltas) / len(uphill_deltas)

        t1 = max(avg_uphill / abs(log(initial_accept_prob)), 1e-12)
        temperature = _temperature_for_step(
            accepted_n,
            t1=t1,
            average_cost_change=avg_abs_delta,
            k=k,
            c=c,
        )
        temperature_history.append(temperature)

        accept_prob = min(1.0, exp(-delta_c / temperature))
        if rng.random() < accept_prob:
            current_positions = neighbor_positions
            current_cost = neighbor_cost
            acceptance_count += 1
            accepted_n += 1
            if current_cost < best_cost:
                best_cost = current_cost
                best_positions = dict(current_positions)

    return FastSAStage1Result(
        accepted_n=accepted_n,
        iterations=iterations,
        acceptance_count=acceptance_count,
        current_cost=current_cost,
        best_cost=best_cost,
        current_positions=current_positions,
        best_positions=best_positions,
        temperature_history=temperature_history,
        delta_history=delta_history,
    )
