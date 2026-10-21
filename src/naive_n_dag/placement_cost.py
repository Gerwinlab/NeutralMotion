"""The placement cost F_time, shared by the FastSA and deterministic placers.

F_time estimates, in microseconds, the two-qubit execution time that depends
on the placement: every AOD load pays its transfers, every stop a Rydberg
pulse, and each load's tour through its stops is timed with the router's
motion law. See COST_FUNCTIONS.md. It replaced the original FastSA objective F
on 2026-10-20 (a copy of F is kept in benchmarks/legacy_f.py).

Layouts are lists indexed by qubit of ``(row, col)`` SLM site indices (grid
coordinates ``(2*row, 2*col)``), or FastSA-style position maps
``{qubit: (x, y)}`` with ``x = 2*col`` and ``y = 2*row``.
"""

from __future__ import annotations

import collections
from typing import Any, Mapping, Sequence

from .dag_helper import _commuting_cz_sections, extract_index_from_bit

# Removed config key; main rejects it with this message.
COST_FUNCTION_REMOVED = (
    "Config key 'cost_function' was removed: placement always uses F_time, the "
    "time-based cost (COST_FUNCTIONS.md). Delete the key."
)


def make_cost(two_qubit_dag, num_qubits: int, config: Mapping[str, Any],
              reference_nodes: Sequence[Any] | None = None) -> "TimeCost":
    """The placement cost for one circuit; built once, evaluated many times."""
    return TimeCost(two_qubit_dag, num_qubits, config, reference_nodes)


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


# ---------------------------------------------------------------------------
# "time": F_time
# ---------------------------------------------------------------------------

def _canonical(v):
    x, y = v
    return (-x, -y) if x < 0 or (x == 0 and y < 0) else (x, y)


class TimeCost:
    """F_time: placement-dependent two-qubit execution time, in microseconds.

    Placement-independent time (single-qubit layers, the fixed number of
    gates) is left out, so only differences between layouts are meaningful.
    Two modes, matching how the scheduler forms its two-qubit layers:

    * fixed layers (``reorder_cz`` off, or no reference nodes): per DAG layer,
      gates are grouped into AOD loads the way ``dynamics._build_groups``
      does (ancilla mover when ``prefer_moving_ancillas``, otherwise either
      operand; a load must fit the AOD span and, unless ``alignment_conc`` is
      -1, the alignment threshold). Each load costs two transfers plus its
      load and unload steps, one pulse per distinct stop, and a
      nearest-neighbour tour from home through its stops and back.
    * CZ sections (``reorder_cz`` on): the post-placement re-layering of each
      commuting CZ section is modelled. Gates are bucketed by signed
      displacement with the reorder's mover rule; a bucket needs as many
      stops as its largest qubit degree (its matchings); the section needs
      at least its largest qubit degree in layers, each one load; and one
      tour visits the section's distinct stops. Other two-qubit gates are
      single-gate layers.

    Not modelled: parity-highway detours and the one-step side offset at a
    stop, Rydberg-radius pulse splitting, and load reuse across layers
    beyond the section tour. Free movers are optimistic: v and -v count as
    one stop, as the router may move either operand.
    """

    name = "time"

    def __init__(self, two_qubit_dag, num_qubits: int, config: Mapping[str, Any],
                 reference_nodes: Sequence[Any] | None = None):
        from .dynamics import _max_grid_spans, _motion_parameters, _step_seconds

        self.num_qubits = num_qubits
        self._motion = {
            "_motion_parameters": config.get("_motion_parameters") or _motion_parameters(config),
            "movement_shape": config.get("movement_shape", "trapezoidal"),
        }
        self._step_seconds = _step_seconds
        self._hop_cache: dict[tuple[int, int], float] = {}
        seconds = lambda key: config[key].to("second").magnitude
        self.t_transfer = seconds("transfer_SLM_AOD")
        self.t_pulse = seconds("average_two_gate_time") + seconds("t_switch")
        # Each load also moves one cardinal step on and one off the SLM row.
        self.load_seconds = 2 * self.t_transfer + 2 * self._hop((1, 0))
        conc = config.get("alignment_conc", 0)
        self.alignment_conc = None if conc is None or conc <= -1 else float(conc)
        self.spans = _max_grid_spans(config["max_dimension"])
        # Fast path, exact: with no alignment threshold and an AOD span that
        # covers the whole array (every config in the repo), every gate of a
        # layer fits the first load, and the mover choice cannot change the
        # stops (a free mover counts v and -v once). So a layer is one load
        # whose stops are its distinct displacements.
        dims = config.get("dimensions")
        self._one_load = (
            self.alignment_conc is None and dims is not None
            and self.spans[0] >= 2 * (int(dims[0]) - 1) and self.spans[1] >= 2 * (int(dims[1]) - 1)
        )

        prefer = bool(config.get("prefer_moving_ancillas", False))
        ancillas = set()
        if reference_nodes is not None:
            ancillas = {extract_index_from_bit(q) for node in reference_nodes
                        if node.op.name == "reset" for q in node.qargs}

        def fixed_mover(a, b):
            if prefer:
                preferred = [q for q in (a, b) if q in ancillas]
                if len(preferred) == 1:
                    return preferred[0]
            return None

        reorder = bool(config.get("reorder_cz", False) or config.get("reorder_cz_by_displacement", False))
        self.mode = "sections" if reorder and reference_nodes is not None else "layers"
        self.units = []                              # (kind, gates, fixed movers)
        if self.mode == "layers":
            for gates in _two_qubit_layers(two_qubit_dag):
                self.units.append(("layer", gates, [fixed_mover(a, b) for a, b in gates]))
        else:
            for is_cz, nodes in _commuting_cz_sections(list(reference_nodes)):
                if is_cz:
                    gates = [(extract_index_from_bit(n.qargs[0]), extract_index_from_bit(n.qargs[1])) for n in nodes]
                    # The reorder's rule: the ancilla when preferred, else the first operand.
                    movers = [fixed_mover(a, b) if fixed_mover(a, b) is not None else a for a, b in gates]
                    self.units.append(("section", gates, movers))
                else:
                    node = nodes[0]
                    if len(node.qargs) == 2 and node.op.name not in ("barrier", "swap"):
                        a, b = (extract_index_from_bit(q) for q in node.qargs)
                        self.units.append(("layer", [(a, b)], [fixed_mover(a, b)]))
        self.layer_pairs = [gates for _, gates, _ in self.units]
        used = {q for gates in self.layer_pairs for gate in gates for q in gate}
        if used and max(used) >= num_qubits:
            raise ValueError(f"The circuit uses qubit {max(used)}, but num_NA is {num_qubits}.")
        # Placement-independent load count of each CZ section (its largest degree).
        self._section_layers = []
        for kind, gates, _ in self.units:
            if kind == "section":
                degree = collections.Counter(q for gate in gates for q in gate)
                self._section_layers.append(max(degree.values()))
            else:
                self._section_layers.append(None)

    def evaluate_positions(self, positions: Mapping[int, tuple[int, int]]) -> float:
        """Evaluate a FastSA position map ``{qubit: (x, y)}`` (x = 2*col, y = 2*row)."""
        return self.evaluate([(positions[q][1] // 2, positions[q][0] // 2) for q in range(self.num_qubits)])

    # -- motion ------------------------------------------------------------
    def _hop(self, delta: tuple[int, int]) -> float:
        """Seconds to move the load by ``delta`` grid steps (router's motion law)."""
        value = self._hop_cache.get(delta)
        if value is None:
            value = self._step_seconds(delta, self._motion)
            self._hop_cache[delta] = value
        return value

    def _tour(self, stops) -> float:
        """Nearest-neighbour tour from home through the stops and back, in seconds."""
        current, total = (0, 0), 0.0
        remaining = list(stops)
        while remaining:
            best_index, best = 0, None
            for index, stop in enumerate(remaining):
                t = self._hop((stop[0] - current[0], stop[1] - current[1]))
                if best is None or t < best:
                    best_index, best = index, t
            total += best
            current = remaining.pop(best_index)
        return total + self._hop((-current[0], -current[1]))

    def star_distance(self, d_row: int, d_col: int) -> float:
        # A single-gate layer is one load out to the stop and back.
        return 2 * self._hop((2 * d_row, 2 * d_col))

    # -- cost ---------------------------------------------------------------
    def _layer_seconds(self, gates, movers, grid):
        if not self._one_load:
            return self._grouped_layer_seconds(gates, movers, grid)
        stops = set()
        for (a, b), fixed in zip(gates, movers):
            mover = a if fixed is None else fixed
            partner = b if mover == a else a
            m, p = grid[mover], grid[partner]
            v = (p[0] - m[0], p[1] - m[1])
            stops.add(v if fixed is not None else _canonical(v))
        ordered = sorted(stops)                      # deterministic tour start order
        return self.load_seconds + self.t_pulse * len(ordered) + self._tour(ordered)

    def _grouped_layer_seconds(self, gates, movers, grid):
        """Group a layer into AOD loads the way ``dynamics._build_groups`` does.

        Each gate joins the first load whose movers still fit the AOD span and
        (with a threshold) whose vectors it aligns with; among its mover
        choices it takes the best-aligned one, the first on ties. Two exact
        shortcuts: a load's bounding box is kept incrementally, and the
        router's alignment score is computed once per gate and load. Swapping
        the mover negates the displacement, which negates the score exactly
        (``_vector_alignment_score`` is odd in its candidate vector).
        """
        from .dynamics import _vector_alignment_score

        row_span, col_span = self.spans
        groups = []          # [min_row, max_row, min_col, max_col, vectors, stop keys]
        for (a, b), fixed in zip(gates, movers):
            choices = []
            for mover in ((fixed,) if fixed is not None else (a, b)):
                partner = b if mover == a else a
                ma, pa = grid[mover], grid[partner]
                choices.append((ma, (pa[0] - ma[0], pa[1] - ma[1])))
            key = (lambda v: v) if fixed is not None else _canonical
            placed = False
            for group in groups:
                fitting = [(site, v) for site, v in choices
                           if max(group[1], site[0]) - min(group[0], site[0]) <= row_span
                           and max(group[3], site[1]) - min(group[2], site[1]) <= col_span]
                if not fitting:
                    continue
                if len(fitting) == 1 and self.alignment_conc is None:
                    site, v = fitting[0]
                else:
                    score = _vector_alignment_score(group[4], fitting[0][1])
                    scored = [(sc, item) for sc, item in zip((score, -score), fitting)
                              if self.alignment_conc is None or sc >= self.alignment_conc]
                    if not scored:
                        continue
                    top = max(sc for sc, _ in scored)
                    site, v = next(item for sc, item in scored if sc == top)
                group[0], group[1] = min(group[0], site[0]), max(group[1], site[0])
                group[2], group[3] = min(group[2], site[1]), max(group[3], site[1])
                group[4].append(v)
                group[5].add(key(v))
                placed = True
                break
            if not placed:
                site, v = choices[0]
                groups.append([site[0], site[0], site[1], site[1], [v], {key(v)}])
        total = 0.0
        for group in groups:
            ordered = sorted(group[5])               # deterministic tour start order
            total += self.load_seconds + self.t_pulse * len(ordered) + self._tour(ordered)
        return total

    def _section_seconds(self, gates, movers, grid, layers):
        buckets = collections.defaultdict(collections.Counter)
        for (a, b), mover in zip(gates, movers):
            partner = b if mover == a else a
            m, p = grid[mover], grid[partner]
            buckets[(p[0] - m[0], p[1] - m[1])].update((a, b))
        stops = sum(max(degree.values()) for degree in buckets.values())
        return (layers * self.load_seconds + self.t_pulse * stops
                + self._tour(sorted(buckets)))

    def evaluate(self, layout: Sequence[tuple[int, int]]) -> float:
        """F_time of an injective layout of ``(row, col)`` site indices, in microseconds."""
        grid = [(2 * r, 2 * c) for r, c in layout]
        total = 0.0
        for (kind, gates, movers), layers in zip(self.units, self._section_layers):
            if kind == "section":
                total += self._section_seconds(gates, movers, grid, layers)
            else:
                total += self._layer_seconds(gates, movers, grid)
        return total * 1e6
