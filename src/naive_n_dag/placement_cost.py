"""The placement cost F_time, shared by the FastSA and deterministic placers.

F_time estimates, in microseconds, the two-qubit execution time that depends
on the placement: every AOD load pays its transfers, every stop a Rydberg
pulse, and each load's tour through its stops is timed with the router's
motion law. It replaced the original FastSA objective F, a copy of which
is kept in Legacy/legacy_f.py.

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
    "time-based cost. Delete the key."
)


_TOUR_CACHE_LIMIT = 200_000


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
# F_time
# ---------------------------------------------------------------------------

def _canonical(v):
    """One representative of ``{v, -v}``, for movers that may go either way."""
    x, y = v
    return (-x, -y) if x < 0 or (x == 0 and y < 0) else (x, y)


class TimeCost:
    """F_time: placement-dependent two-qubit execution time, in microseconds (differences only).

    Each two-qubit layer (with ``reorder_cz``, each layer
    ``reorder_cz_layers_by_displacement`` will build) is grouped into AOD
    loads like ``dynamics._build_groups``; a load pays two transfers, one
    pulse per distinct displacement, and a nearest-neighbour highway tour
    priced with the router's side DP. Not modelled: load reuse across layers,
    Rydberg-radius pulse splitting, the router's look-ahead mover choice and
    one-atom-per-load fallback, and its exact tour order.
    """

    def __init__(self, two_qubit_dag, num_qubits: int, config: Mapping[str, Any],
                 reference_nodes: Sequence[Any] | None = None):
        from .dynamics import _CARDINAL, _max_grid_spans, _motion_parameters, _route, _step_seconds

        self.num_qubits = num_qubits
        motion = {
            "_motion_parameters": config.get("_motion_parameters") or _motion_parameters(config),
            "movement_shape": config.get("movement_shape", "trapezoidal"),
        }
        self._route_seconds = lambda a, b: _route(a, b, motion)[0]
        self._cardinal = _CARDINAL
        self._transfer_seconds = tuple(_step_seconds(step, motion) for step in _CARDINAL)
        # Side tables per displacement (see _sides); they replace 16 route
        # lookups per stop in the side DP. Bounded by the array's displacements.
        self._side_cache: dict[tuple[int, int], tuple] = {}
        self._entry_cache: dict[tuple[int, int], tuple] = {}
        # Tour seconds per stop tuple. A search move changes only the layers
        # touching the moved qubits, so most loads repeat their stops between
        # evaluations. Cleared when full, which bounds memory.
        self._tour_cache: dict[tuple, float] = {}
        seconds = lambda key: config[key].to("second").magnitude
        self.t_transfer = seconds("transfer_SLM_AOD")
        self.t_pulse = seconds("average_two_gate_time") + seconds("t_switch")
        # The load/unload steps' motion is part of the tour's first and last legs.
        self.load_seconds = 2 * self.t_transfer
        conc = config.get("alignment_conc", 0)
        self.alignment_conc = None if conc is None or conc <= -1 else float(conc)
        self.spans = _max_grid_spans(config["max_dimension"])
        # Fast path, exact: with no alignment threshold and an AOD span that
        # covers the whole array, every gate of a layer fits the first load,
        # and the mover choice cannot change the stops (a free mover counts v
        # and -v once). So a layer is one load whose stops are its distinct
        # displacements.
        dims = config.get("dimensions")
        self._one_load = (
            self.alignment_conc is None and dims is not None
            and self.spans[0] >= 2 * (int(dims[0]) - 1) and self.spans[1] >= 2 * (int(dims[1]) - 1)
        )

        prefer = bool(config.get("prefer_moving_ancillas", False))
        ancillas = config.get("_ancilla_qubits", frozenset())

        def fixed_mover(a, b):
            """The router's mover when it is forced: the gate's only ancilla."""
            if prefer:
                preferred = [q for q in (a, b) if q in ancillas]
                if len(preferred) == 1:
                    return preferred[0]
            return None

        reorder = bool(config.get("reorder_cz", False) or config.get("reorder_cz_by_displacement", False))
        self.mode = "sections" if reorder and reference_nodes is not None else "layers"
        # ("layer", gates, fixed movers) or ("section", gates, fixed movers, bucket movers, nodes)
        self.units = []
        if self.mode == "layers":
            for gates in _two_qubit_layers(two_qubit_dag):
                self.units.append(("layer", gates, [fixed_mover(a, b) for a, b in gates]))
        else:
            for is_cz, nodes in _commuting_cz_sections(list(reference_nodes)):
                if is_cz:
                    gates = [(extract_index_from_bit(n.qargs[0]), extract_index_from_bit(n.qargs[1])) for n in nodes]
                    fixed = [fixed_mover(a, b) for a, b in gates]
                    # The reorder's bucketing rule: the ancilla when preferred, else the first operand.
                    bucket = [m if m is not None else a for m, (a, b) in zip(fixed, gates)]
                    self.units.append(("section", gates, fixed, bucket, list(nodes)))
                else:
                    node = nodes[0]
                    if len(node.qargs) == 2 and node.op.name != "barrier":
                        a, b = (extract_index_from_bit(q) for q in node.qargs)
                        self.units.append(("layer", [(a, b)], [fixed_mover(a, b)]))
        # Operands per gate as small ints (one per distinct Qubit object, so
        # disjointness matches the reorder's node.qargs test), and colourings
        # per bucket: a search move changes few buckets, and _color_cz_block
        # depends only on the bucket's gates.
        bit_ids: dict[Any, int] = {}
        self._operands = [
            [frozenset(bit_ids.setdefault(q, len(bit_ids)) for q in node.qargs) for node in unit[4]]
            if unit[0] == "section" else None
            for unit in self.units
        ]
        self._coloring_cache: dict[tuple, tuple] = {}
        self.layer_pairs = [unit[1] for unit in self.units]
        used = {q for gates in self.layer_pairs for gate in gates for q in gate}
        if used and max(used) >= num_qubits:
            raise ValueError(f"The circuit uses qubit {max(used)}, but num_NA is {num_qubits}.")
        # Incremental evaluation (see evaluate): which units and gates each qubit touches.
        self._qubit_units = [[] for _ in range(num_qubits)]
        self._qubit_gates = []
        for unit_index, gates in enumerate(self.layer_pairs):
            by_qubit = collections.defaultdict(list)
            for gate_index, gate in enumerate(gates):
                for q in set(gate):
                    by_qubit[q].append(gate_index)
            for q in by_qubit:
                self._qubit_units[q].append(unit_index)
            self._qubit_gates.append(dict(by_qubit))
        self._last_grid: list | None = None
        self._last_unit_seconds: list[tuple[float, ...]] = []
        self._last_layer_seconds: list[dict[tuple, float]] = []

    def evaluate_positions(self, positions: Mapping[int, tuple[int, int]]) -> float:
        """Evaluate a FastSA position map ``{qubit: (x, y)}`` (x = 2*col, y = 2*row)."""
        return self.evaluate([(positions[q][1] // 2, positions[q][0] // 2) for q in range(self.num_qubits)])

    # -- motion ------------------------------------------------------------
    def _sides(self, delta: tuple[int, int]) -> tuple:
        """Route seconds between the four sides of two stops ``delta`` apart.

        Entry ``4*s + t`` is side ``s`` to side ``t``; the last entry is their
        minimum. One table serves every pair of stops at that displacement.
        """
        table = self._side_cache.get(delta)
        if table is None:
            route, dr, dc = self._route_seconds, delta[0], delta[1]
            table = tuple(route(s, (dr + t[0], dc + t[1])) for s in self._cardinal for t in self._cardinal)
            table += (min(table),)
            self._side_cache[delta] = table
        return table

    def _entry(self, stop: tuple[int, int]) -> tuple:
        """Seconds from home to each side of ``stop`` (best transfer step, then the route); minimum last."""
        value = self._entry_cache.get(stop)
        if value is None:
            sides, transfer = self._sides(stop), self._transfer_seconds
            value = tuple(min(transfer[s] + sides[4 * s + t] for s in range(4)) for t in range(4))
            value += (min(value),)
            self._entry_cache[stop] = value
        return value

    def _leg(self, delta: tuple[int, int]) -> float:
        """Seconds between stops ``delta`` grid steps apart, minimised over sides.

        The load sits beside each stop's partner, so the leg is the best
        parity-highway route between a site beside one stop and a site beside
        the other.
        """
        return self._sides(delta)[16]

    def _end(self, stop: tuple[int, int]) -> float:
        """Seconds between home and a stop: one transfer step, then a highway route beside it."""
        return self._entry(stop)[4]

    def _tour(self, stops) -> float:
        """Seconds for one load: home, through the stops, and back.

        Nearest-neighbour order on side-relaxed legs, priced with the side DP
        of ``dynamics._optimize_visit_sides`` on the cached tables (same float,
        no routes built).
        """
        key = tuple(stops)
        cached = self._tour_cache.get(key)
        if cached is not None:
            return cached
        if not key:
            return 0.0
        sides, entry, cache = self._sides, self._entry, self._side_cache
        remaining = list(key)
        best_index = min(range(len(remaining)), key=lambda i: entry(remaining[i])[4])
        previous = remaining.pop(best_index)
        c0, c1, c2, c3 = entry(previous)[:4]
        while remaining:
            # Nearest neighbour, first on ties; inlined, as it is the hot loop.
            pr, pc = previous
            best, best_index = None, 0
            for index, (r, c) in enumerate(remaining):
                table = cache.get((r - pr, c - pc)) or sides((r - pr, c - pc))
                if best is None or table[16] < best:
                    best, best_index = table[16], index
            current = remaining.pop(best_index)
            m = sides((current[0] - pr, current[1] - pc))
            c0, c1, c2, c3 = (
                min(c0 + m[0], c1 + m[4], c2 + m[8], c3 + m[12]),
                min(c0 + m[1], c1 + m[5], c2 + m[9], c3 + m[13]),
                min(c0 + m[2], c1 + m[6], c2 + m[10], c3 + m[14]),
                min(c0 + m[3], c1 + m[7], c2 + m[11], c3 + m[15]),
            )
            previous = current
        m, transfer = sides((-previous[0], -previous[1])), self._transfer_seconds
        value = min(cost + m[4 * s + e] + transfer[e]
                    for s, cost in enumerate((c0, c1, c2, c3)) for e in range(4))
        if len(self._tour_cache) >= _TOUR_CACHE_LIMIT:
            self._tour_cache.clear()
        self._tour_cache[key] = value
        return value

    def star_distance(self, d_row: int, d_col: int) -> float:
        # A single-gate layer is one load out to the stop and back.
        return 2 * self._end((2 * d_row, 2 * d_col))

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
        """Group a layer into AOD loads like ``dynamics._build_groups``, without its look-ahead mover choice.

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
            if fixed is not None and self.alignment_conc is None:
                # One mover and no threshold: the first load it fits (hot path, inlined).
                (r, c), (pr, pc) = grid[fixed], grid[b if fixed == a else a]
                v = (pr - r, pc - c)
                for group in groups:
                    r0, r1, c0, c1 = group[0], group[1], group[2], group[3]
                    if r < r0:
                        r0 = r
                    elif r > r1:
                        r1 = r
                    if c < c0:
                        c0 = c
                    elif c > c1:
                        c1 = c
                    if r1 - r0 <= row_span and c1 - c0 <= col_span:
                        group[0], group[1], group[2], group[3] = r0, r1, c0, c1
                        group[4].append(v)
                        group[5].add(v)
                        break
                else:
                    groups.append([r, r, c, c, [v], {v}])
                continue
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

    def _section_layers(self, unit_index, grid):
        """The two-qubit layers the reorder step builds for one CZ section.

        Mirrors ``dag_helper.reorder_cz_layers_by_displacement``: bucket by
        signed displacement, split each bucket into matchings with
        ``_color_cz_block``, sort them largest first and pack them first fit
        into qubit-disjoint layers. Returns gate indices per layer.
        """
        from .dag_helper import _color_cz_block

        _, gates, _, bucket_movers, nodes = self.units[unit_index]
        operands = self._operands[unit_index]
        buckets = {}
        for i, ((a, b), mover) in enumerate(zip(gates, bucket_movers)):
            partner = b if mover == a else a
            m, p = grid[mover], grid[partner]
            buckets.setdefault((p[0] - m[0], p[1] - m[1]), []).append(i)
        bundles = []
        for members in buckets.values():
            key = (unit_index, *members)
            coloring = self._coloring_cache.get(key)
            if coloring is None:
                index = {id(nodes[i]): i for i in members}
                coloring = tuple(tuple(index[id(node)] for node in matching)
                                 for matching in _color_cz_block([nodes[i] for i in members]))
                if len(self._coloring_cache) >= _TOUR_CACHE_LIMIT:
                    self._coloring_cache.clear()
                self._coloring_cache[key] = coloring
            bundles.extend(coloring)
        bundles.sort(key=lambda matching: -len(matching))
        packed = []
        for matching in bundles:
            used_here = frozenset().union(*(operands[i] for i in matching))
            for layer, used in packed:
                if used_here.isdisjoint(used):
                    layer.extend(matching)
                    used |= used_here
                    break
            else:
                packed.append((list(matching), set(used_here)))
        return [layer for layer, _ in packed]

    def evaluate(self, layout: Sequence[tuple[int, int]]) -> float:
        """F_time (microseconds) of an injective layout of ``(row, col)`` sites.

        Only units touching qubits that moved since the last call are
        re-priced; the result depends only on ``layout``, not the call history.
        """
        grid = [(2 * r, 2 * c) for r, c in layout]
        last = self._last_grid
        changed = None
        if last is not None and len(last) == len(grid):
            changed = [q for q, (old, new) in enumerate(zip(last, grid)) if old != new]
            if 4 * len(changed) > len(grid):          # a fresh layout: re-price everything
                changed = None
        if changed is None:
            dirty_units = range(len(self.units))
            self._last_unit_seconds = [()] * len(self.units)
            self._last_layer_seconds = [{} for _ in self.units]
        else:
            dirty_units = sorted({u for q in changed for u in self._qubit_units[q]})
        for unit_index in dirty_units:
            unit = self.units[unit_index]
            if unit[0] == "layer":
                self._last_unit_seconds[unit_index] = (self._layer_seconds(unit[1], unit[2], grid),)
                continue
            gates, fixed = unit[1], unit[2]
            previous = self._last_layer_seconds[unit_index]
            moved = set()
            if changed is not None:
                by_qubit = self._qubit_gates[unit_index]
                for q in changed:
                    moved.update(by_qubit.get(q, ()))
            seconds, layer_seconds = [], {}
            for layer in self._section_layers(unit_index, grid):
                key = tuple(layer)
                value = previous.get(key) if moved.isdisjoint(key) else None
                if value is None:
                    value = self._layer_seconds([gates[i] for i in layer], [fixed[i] for i in layer], grid)
                layer_seconds[key] = value
                seconds.append(value)
            self._last_unit_seconds[unit_index] = tuple(seconds)
            self._last_layer_seconds[unit_index] = layer_seconds
        self._last_grid = grid
        total = 0.0
        for seconds in self._last_unit_seconds:
            for value in seconds:
                total += value
        return total * 1e6
