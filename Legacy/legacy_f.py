"""Legacy placement cost F (the original FastSA objective), kept for reference.

F was the placement objective until 2026-10-20, when F_time
(``naive_n_dag.placement_cost``) replaced it: tuned FastSA and the
deterministic placer both give faster schedules with F_time (see
cost_fidelity.py and fastsa_tuning.py in this folder). F is no longer in
``src/``; this copy lets those benchmarks reproduce the comparison:

  compute_fastsa_cost   the original pure-Python objective
  FastSACost            the same values, precompiled and vectorised. Pass an
                        instance as ``cost=`` to ``run_fastsa`` or
                        ``run_deterministic_placement`` to optimise F.

    F = (1/L) * sum_l ( (1/|D_l|) * sum_{distinct +-v} |v| / count(v)
                        - mean over gate pairs of |cos| * min/max(|u|, |v|) )
"""

from __future__ import annotations

import math
from collections import Counter
from math import hypot
from typing import Iterable, Mapping, Sequence

import numpy as np

from naive_n_dag.dag_helper import extract_index_from_bit


def qubit_position_map(qubits):
    """Map logical qubit id to integer grid coordinate (x, y) = (col, row)."""
    pos = {}
    for qubit in qubits:
        row, col = qubit.grid_position()
        pos[qubit.id] = (col, row)
    return pos


def vector_alignment_score(vectors: Iterable[tuple[int, int]]) -> float:
    """Return mean pairwise alignment score in [0, 1] for movement vectors."""
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
            normalizer += 1
            score += cosine * mag_ratio
    return score / normalizer


def compute_fastsa_cost(
    two_qubit_layers: Iterable[Iterable["DAGOpNode"]],
    positions_or_qubits,
    normalize_by_length: float = 1.0,
) -> tuple[float, list[dict]]:
    """Compute F for a placement; return ``(F, per-layer stats)``.

    ``two_qubit_layers`` is the two-qubit DAGCircuit (its ``layers()`` are
    used); ``positions_or_qubits`` is a ``{qubit: (x, y)}`` map or a list of
    placed Qubit objects.
    """
    if isinstance(positions_or_qubits, Mapping):
        positions = dict(positions_or_qubits)
    else:
        positions = qubit_position_map(positions_or_qubits)

    def canonical(v: tuple[int, int]) -> tuple[int, int]:
        x, y = v
        if x < 0 or (x == 0 and y < 0):
            return (-x, -y)
        return (x, y)

    gate_layers = [
        [node for node in layer["graph"].op_nodes()
         if len(node.qargs) == 2 and node.op.name != "barrier"]
        for layer in two_qubit_layers.layers()
    ]
    gate_layers = [layer for layer in gate_layers if layer]
    total_cost = 0.0
    normalize_by_layers = max(len(gate_layers), 1)
    layer_stats_list: list[dict] = []
    for layer_nodes in gate_layers:
        vectors: list[tuple[int, int]] = []
        layer_cost = 0.0
        for node in layer_nodes:
            q0 = extract_index_from_bit(node.qargs[0])
            q1 = extract_index_from_bit(node.qargs[1])
            if q0 not in positions or q1 not in positions:
                raise KeyError(f"Missing qubit positions for q[{q0}] or q[{q1}].")

            x0, y0 = positions[q0]
            x1, y1 = positions[q1]
            vectors.append((x1 - x0, y1 - y0))

        alignment = vector_alignment_score(vectors)

        counts = Counter(canonical(v) for v in vectors)
        norm_counts = len(counts)
        for vec, count in counts.items():
            mag = hypot(vec[0], vec[1])
            layer_cost += mag * (1.0 / count) / normalize_by_length / norm_counts
        total_cost += (layer_cost - alignment) / normalize_by_layers
        layer_stats_list.append({
            "vectors": vectors,
            "vector_alignment_score": alignment,
            "layer_cost": layer_cost,
        })
    return total_cost, layer_stats_list


def _two_qubit_layers(two_qubit_dag):
    """Non-empty layers of (q0, q1) for non-barrier two-qubit gates."""
    layers = []
    for layer in two_qubit_dag.layers():
        gates = [
            (extract_index_from_bit(node.qargs[0]), extract_index_from_bit(node.qargs[1]))
            for node in layer["graph"].op_nodes()
            if len(node.qargs) == 2 and node.op.name != "barrier"
        ]
        if gates:
            layers.append(gates)
    return layers


class FastSACost:
    """``compute_fastsa_cost`` for one two-qubit DAG, precompiled and vectorised.

    Matches ``compute_fastsa_cost(dag, positions, normalize_by_length=1)``
    above: layers keep their non-barrier two-qubit gates,
    empty layers are dropped, and the sum is divided by the number of
    non-empty layers. Two exact rewrites make it fast: the layer structure is
    derived once (the original rebuilds ``dag.layers()`` on every call), and
    ``|cos| * min/max == |u.v| / max(|u|^2, |v|^2)`` removes every square root
    from the pairwise alignment term.
    """

    name = "fastsa"

    def __init__(self, two_qubit_dag, num_qubits: int):
        layers = _two_qubit_layers(two_qubit_dag)
        self.layer_pairs = layers
        self.n_layers = max(len(layers), 1)
        used = {q for gates in layers for gate in gates for q in gate}
        if used and max(used) >= num_qubits:
            raise ValueError(f"The circuit uses qubit {max(used)}, but num_NA is {num_qubits}.")
        self.num_qubits = num_qubits
        q0, q1, gate_layer, pi, pj, pair_layer = [], [], [], [], [], []
        base = 0
        for index, gates in enumerate(layers):
            for a, b in gates:
                q0.append(a)
                q1.append(b)
                gate_layer.append(index)
            m = len(gates)
            for i in range(m):
                for j in range(i + 1, m):
                    pi.append(base + i)
                    pj.append(base + j)
                    pair_layer.append(index)
            base += m
        self.q0 = np.asarray(q0, dtype=np.int64)
        self.q1 = np.asarray(q1, dtype=np.int64)
        self.gate_layer = np.asarray(gate_layer, dtype=np.int64)
        self.pi = np.asarray(pi, dtype=np.int64)
        self.pj = np.asarray(pj, dtype=np.int64)
        pair_layer = np.asarray(pair_layer, dtype=np.int64)
        # Pairs are emitted layer by layer, so a segmented reduceat sums them.
        self.pair_layers = np.flatnonzero(np.bincount(pair_layer, minlength=len(layers)) > 0)
        self.pair_starts = np.searchsorted(pair_layer, self.pair_layers).astype(np.intp)
        per_layer = np.bincount(self.gate_layer, minlength=len(layers))
        self.pairs_per_layer = (per_layer * (per_layer - 1) // 2).astype(np.float64)
        self.n_gates = len(q0)
        self._positions = np.zeros((num_qubits, 2), dtype=np.int64)

    def evaluate(self, layout: Sequence[tuple[int, int]]) -> float:
        """Objective of an injective layout of ``(row, col)`` site indices."""
        if not self.n_gates:
            return 0.0
        sites = np.asarray(layout, dtype=np.int64)
        positions = self._positions
        positions[:len(sites), 0] = 2 * sites[:, 1]          # x = 2 * col
        positions[:len(sites), 1] = 2 * sites[:, 0]          # y = 2 * row
        vectors = positions[self.q1] - positions[self.q0]
        vx, vy = vectors[:, 0], vectors[:, 1]
        n2 = vx * vx + vy * vy
        n_layers = len(self.pairs_per_layer)

        align = np.zeros(n_layers)
        if self.pi.size:
            dot = np.abs(vx[self.pi] * vx[self.pj] + vy[self.pi] * vy[self.pj])
            inverse = 1.0 / n2
            term = dot * np.minimum(inverse[self.pi], inverse[self.pj])
            align[self.pair_layers] = np.add.reduceat(term, self.pair_starts)
        ppl = self.pairs_per_layer
        alignment = np.divide(align, ppl, out=np.zeros_like(align), where=ppl > 0)

        # Distinct canonical vectors per layer (v and -v are one class).
        flip = (vx < 0) | ((vx == 0) & (vy < 0))
        cx = np.where(flip, -vx, vx)
        cy = np.where(flip, -vy, vy)
        span = int(max(np.abs(cx).max(), np.abs(cy).max())) + 1
        key = self.gate_layer * (4 * span * span) + (cx + span) * (2 * span) + (cy + span)
        order = np.argsort(key, kind="stable")
        sorted_keys = key[order]
        first = np.empty(sorted_keys.size, dtype=bool)
        first[0] = True
        np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=first[1:])
        starts = np.flatnonzero(first)
        counts = np.diff(np.append(starts, sorted_keys.size))
        unique_layer = self.gate_layer[order][starts]
        distinct = np.bincount(unique_layer, minlength=n_layers).astype(np.float64)
        contribution = np.sqrt(n2[order][starts].astype(np.float64)) / counts / distinct[unique_layer]
        layer_cost = np.bincount(unique_layer, weights=contribution, minlength=n_layers)
        return float(np.sum(layer_cost - alignment) / self.n_layers)

    def evaluate_positions(self, positions):
        """Evaluate a FastSA position map ``{qubit: (x, y)}`` (x = 2*col, y = 2*row)."""
        return self.evaluate([(positions[q][1] // 2, positions[q][0] // 2) for q in range(self.num_qubits)])

    def star_distance(self, d_row, d_col):
        """Per-gate cost of a single-gate layer at this site offset (closed forms)."""
        return math.hypot(2 * d_row, 2 * d_col)
