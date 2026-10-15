from __future__ import annotations

import contextlib
import math
import os
import re
import sys
import tempfile
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import _is_reorderable_cz, format_node_line, op_node_signature
from .grid import Qubit

# Rydberg blockade radius in meters (15 micrometers). Lattice geometry comes from
# the configured lattice_spacing; this radius only decides which two-qubit
# gates may share one Rydberg pulse.
RYDBERG_RADIUS = 15e-6

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


# Per-solve time budget (seconds) for fast_tsp's local search; config key tsp_time_limit.
_DEFAULT_TSP_TIME_LIMIT = 0.05
# Concorde returns no tour for instances smaller than this; they go to fast_tsp.
_CONCORDE_MIN_NODES = 5
# Fixed Concorde seed: 0 would seed from the clock and break ties differently per run.
_CONCORDE_SEED = 1


_SIDE_ORDERING_REMOVED = (
    "Config key '_side_ordering' was removed: the greedy side ordering no longer exists, and "
    "every group uses the side-priced TSP tour (Concorde, else fast_tsp). Delete the key."
)

_PARALLEL_REMOVED = (
    "Config key 'parallel' was removed: gates always share AOD loads when the AOD "
    "size allows. To move one atom at a time, set max_dimension to [1, 1]. Delete the key."
)


class _TourSolverError(RuntimeError):
    """No TSP backend produced a tour for this instance."""


@dataclass(frozen=True)
class _LayeredTourGraph:
    """Symmetric TSP instance for one AOD load that stays loaded across layers.

    Node 0 is the load terminal at the current rigid offset; the last node is
    the unload terminal at home (0, 0). Nodes in between are the distinct
    ``cities`` in order, each a (layer, displacement) pair. Terminals count as
    layers -1 and ``layer_count``, closing a ring of layers with one free
    unload-to-load edge. Only edges within a layer or between consecutive
    layers exist; every layer crossing pays ``penalty``. Missing edges are
    ``math.inf`` (an integer solver such as Concorde needs a finite stand-in).
    """

    cities: tuple[tuple[int, tuple[int, int]], ...]
    node_layers: tuple[int, ...]
    weights: tuple[tuple[float, ...], ...]
    penalty: float


def _layered_tour_graph(cities, config, start_offset):
    """Build the penalized layer-ring graph for labelled (layer, displacement) cities.

    A Hamiltonian cycle crosses every cut an even number of times. With the
    unload-to-load edge, each of the layer_count + 1 boundaries is crossed at
    least once; without it, at least twice. Any tour that splits a layer or
    returns to an earlier one therefore pays two or more extra penalties. The
    penalty exceeds a complete feasible tour's movement, so such tours always
    lose: an exact solver returns a layer-ordered tour, and the movement cost
    is the tour cost minus (layer_count + 1) penalties.

    Edges are priced by ``_side_movement``, the cheapest interaction sides of
    the two cities, so there is one node per city and the matrix stays
    symmetric. That needs even-even targets and a load start at home or an
    AOD site.
    """
    if not all(r % 2 == 0 and c % 2 == 0 for _, (r, c) in cities) or not (
        start_offset == (0, 0) or _is_valid_aod_position(start_offset)
    ):
        raise _TourSolverError(
            "Side-priced tours need even-even targets and a load start at home or an AOD site."
        )
    layer_count = max(layer for layer, _ in cities) + 1
    nodes = [(-1, start_offset), *cities, (layer_count, (0, 0))]
    end = len(nodes) - 1
    movement = _side_movement(nodes, config)

    ordered = [0, *sorted(range(1, end), key=lambda node: nodes[node][0]), end]
    bound = sum(movement(a, b) for a, b in zip(ordered, ordered[1:]))
    penalty = bound + max(bound, 1e-12)
    weights = tuple(
        tuple(
            0.0 if i == j or {i, j} == {0, end}
            else math.inf if abs(a - b) > 1
            else movement(i, j) + penalty * (a != b)
            for j, (b, _) in enumerate(nodes)
        )
        for i, (a, _) in enumerate(nodes)
    )
    return _LayeredTourGraph(
        tuple(cities), tuple(layer for layer, _ in nodes), weights, penalty,
    )


def _side_movement(nodes, config):
    """Cheapest parity-routed motion between any interaction sides of two nodes.

    A city is visited from one of four cardinal sites around its even-even
    target. The load terminal starts at its offset, or at home with any
    transfer direction when that offset is (0, 0); the unload terminal is
    reached by routing to any home-adjacent site and transferring down. The
    minimum is symmetric because route costs are, so the graph keeps one node
    per city. Leaving and arriving sides may differ, so this prices order
    only; ``_optimize_visit_sides`` then chooses consistent sides exactly.
    """
    end = len(nodes) - 1
    transfers = [(_step_seconds(step, config), step) for step in _CARDINAL]
    start = nodes[0][1]
    entries = transfers if start == (0, 0) else [(0.0, start)]
    sites = [[(0.0, (r + dr, c + dc)) for dr, dc in _CARDINAL] for _, (r, c) in nodes]

    def route(a, b):
        return _route(a, b, config)[0]

    @lru_cache(maxsize=None)
    def movement(i, j):
        if i > j:
            return movement(j, i)
        first = entries if i == 0 else sites[i]
        second = transfers if j == end else sites[j]
        return min(a + route(p, q) + b for a, p in first for b, q in second)
    return movement


def _route(start, end, config):
    """Parity-routed (motion seconds, steps) from ``start`` to ``end``, cached.

    Highways depend only on coordinate parity, so a route is fixed by the
    start's parity class and the displacement; translating both endpoints by
    an even vector leaves it unchanged. The cache is shared across groups.
    """
    return _route_steps(
        (start[0] % 2, start[1] % 2), (end[0] - start[0], end[1] - start[1]),
        config.get("_motion_parameters") or _motion_parameters(config),
        config.get("movement_shape", "trapezoidal"),
    )


@lru_cache(maxsize=1 << 16)
def _route_steps(parity, delta, motion_parameters, movement_shape):
    config = {"_motion_parameters": motion_parameters, "movement_shape": movement_shape}
    steps = tuple(parity_route_moves(parity, (parity[0] + delta[0], parity[1] + delta[1]), config))
    return sum(_step_seconds(step, config) for step in steps), steps


def _integer_weights(weights, finite_max, missing):
    """Scale finite weights to integers in [0, finite_max]; absent edges become ``missing``."""
    top = max((w for row in weights for w in row if math.isfinite(w)), default=0.0)
    scale = finite_max / top if top > 0 else 1.0
    return [[round(w * scale) if math.isfinite(w) else missing for w in row] for row in weights]


@contextlib.contextmanager
def _silenced_output():
    """Discard output on file descriptors 1 and 2, including from C code.

    Concorde and its LP solver print progress and recoverable diagnostics
    there; failures still surface through the returned solution.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


@lru_cache(maxsize=None)
def _tsp_backends():
    """Import the TSP backends once, warning about each one that is missing."""
    try:
        from concorde.problem import Problem
        from concorde.tsp import TSPSolver
        concorde = (Problem, TSPSolver)
    except ImportError as exc:
        concorde = None
        warnings.warn(
            f"pyconcorde (Concorde) is unavailable ({exc}); displacement tours use fast_tsp instead.",
            RuntimeWarning, stacklevel=2,
        )
    try:
        import fast_tsp
    except ImportError as exc:
        fast_tsp = None
        warnings.warn(
            f"fast_tsp is unavailable ({exc}); tours it would solve fall back to alignment ordering.",
            RuntimeWarning, stacklevel=2,
        )
    return concorde, fast_tsp


def _concorde_tour(weights, problem_type, solver_type):
    """Exact tour from Concorde; integer weights keep any absent edge above every feasible tour.

    Concorde writes branch-and-bound restart files (``*.res``) into the working
    directory, so the solve runs inside a temporary directory that is removed.
    """
    n = len(weights)
    finite_max = max(1, min(10**6, (2**31 - 1) // (n * n + n)))
    matrix = _integer_weights(weights, finite_max, n * finite_max + 1)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tour.tsp")
        problem_type.from_matrix(matrix).to_tsp(path)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with _silenced_output():
                solution = solver_type.from_tspfile(path).solve(verbose=False, random_seed=_CONCORDE_SEED)
        finally:
            os.chdir(cwd)
    if not solution.found_tour:
        raise _TourSolverError("Concorde did not find a tour.")
    return [int(node) for node in solution.tour]


def _fast_tsp_tour(weights, fast_tsp, time_limit):
    """Local-search tour from fast_tsp, which takes integer weights up to 65535."""
    n = len(weights)
    matrix = _integer_weights(weights, max(1, 65534 // n), 65535)
    return [int(node) for node in fast_tsp.find_tour(matrix, time_limit)]


def _tour_solvers(node_count, config):
    """Solvers to try, in order: Concorde (installed, large enough), then fast_tsp."""
    concorde, fast_tsp = _tsp_backends()
    solvers = []
    if concorde is not None and node_count >= _CONCORDE_MIN_NODES:
        solvers.append(("Concorde", lambda weights: _concorde_tour(weights, *concorde)))
    if fast_tsp is not None:
        limit = config.get("tsp_time_limit", _DEFAULT_TSP_TIME_LIMIT)
        solvers.append(("fast_tsp", lambda weights: _fast_tsp_tour(weights, fast_tsp, limit)))
    return solvers


def _solve_tour(weights, config, accept=lambda tour: tour):
    """First accepted Hamiltonian cycle of a symmetric matrix from the solver chain.

    Concorde is tried first; fast_tsp is used when pyconcorde is missing, the
    graph is too small for Concorde, or Concorde fails. ``accept`` validates a
    tour (and may convert it); a solver error or a rejected tour moves on to the
    next solver. Absent edges (``math.inf``) become large finite weights.
    """
    failures = []
    for name, solve in _tour_solvers(len(weights), config):
        try:
            result = accept(solve(weights))
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            continue
        if failures:
            warnings.warn(
                f"{'; '.join(failures)}; used {name} instead.", RuntimeWarning, stacklevel=2,
            )
        return result
    raise _TourSolverError("; ".join(failures) or "No TSP solver is available for this instance.")


def _validated_layer_order(tour, graph):
    """Return city indices of a solver's tour after checking it is faithful.

    Exact solvers cannot violate the layer ring, but approximate ones can. The
    tour must be a Hamiltonian cycle over existing edges that closes through
    the unload-to-load edge and, oriented from loading, never revisits a layer.
    """
    tour = list(tour)
    count = len(graph.weights)
    end = count - 1
    if len(tour) != count or set(tour) != set(range(count)):
        raise ValueError("TSP solver did not return a Hamiltonian tour.")
    if any(not math.isfinite(graph.weights[a][b]) for a, b in zip(tour, tour[1:] + tour[:1])):
        raise ValueError("TSP solver tour uses an edge absent from the layer graph.")
    start = tour.index(0)
    cycle = tour[start:] + tour[:start]
    if cycle[1] == end:
        cycle = [0, *reversed(cycle[1:])]
    layers = [graph.node_layers[node] for node in cycle]
    if cycle[-1] != end or layers != sorted(layers):
        raise ValueError("Displacement tour violated layer order.")
    return [node - 1 for node in cycle[1:-1]]


def _displacement_layer_labels(vectors, layer_ids):
    """Normalize chronological labels, allowing empty intervening DAG layers."""
    if layer_ids is None:
        return [0] * len(vectors)
    if len(layer_ids) != len(vectors) or any(
        not isinstance(label, int) or isinstance(label, bool) or label < 0 for label in layer_ids
    ):
        raise ValueError("Expected one nonnegative integer layer ID per displacement.")
    ranks = {label: rank for rank, label in enumerate(sorted(set(layer_ids)))}
    return [ranks[label] for label in layer_ids]


def _alignment_order(vectors, labels, start_offset):
    """Fallback order: alignment heuristic within each layer, layers in order."""
    order = []
    for label in sorted(set(labels)):
        buckets = {}
        for i, stage in enumerate(labels):
            if stage == label:
                buckets.setdefault(vectors[i], []).append(i)
        unique = list(buckets)
        local = _sort_group_by_alignment([
            (vector[0] - start_offset[0], vector[1] - start_offset[1]) for vector in unique
        ])
        order.extend(i for index in local for i in buckets[unique[index]])
    return order


def _order_group_displacements(
    vectors: list[tuple[int, int]], config: dict,
    *, start_offset: tuple[int, int] = (0, 0),
    layer_ids: list[int] | None = None,
    tour_solver=None,
) -> list[int]:
    """Return all vector indices in a side-priced TSP displacement tour.

    Cities are partner positions relative to each mover's immutable home,
    labelled (layer, displacement) on the penalized ring of
    ``_layered_tour_graph``; the path starts at the current rigid-load offset
    and ends at home (0, 0). Edges use the cheapest interaction sides, and
    ``_optimize_visit_sides`` then fixes sides exactly for the chosen order.
    Repeated displacements within a layer expand consecutively in input order.

    ``tour_solver`` maps the weight matrix to a Hamiltonian cycle; it defaults
    to ``_solve_tour`` (Concorde, then fast_tsp if Concorde is missing, too
    small for the graph, or fails). Only when no solver yields a valid
    layer-ordered tour does the alignment heuristic order the targets instead;
    the two methods are never both run and compared.
    """
    labels = _displacement_layer_labels(vectors, layer_ids)
    if not vectors:
        return []
    indices_by_city = {}
    for index, city in enumerate(zip(labels, vectors)):
        indices_by_city.setdefault(city, []).append(index)
    cities = list(indices_by_city)
    if len(cities) == 1:
        return indices_by_city[cities[0]]
    try:
        graph = _layered_tour_graph(cities, config, start_offset)

        def accept(tour):
            return _validated_layer_order(tour, graph)

        if tour_solver is None:
            city_order = _solve_tour(graph.weights, config, accept)
        else:
            city_order = accept(tour_solver(graph.weights))
        order = [cities[city] for city in city_order]
    except Exception as exc:
        warnings.warn(
            f"Displacement TSP failed ({exc}); falling back to alignment ordering.",
            RuntimeWarning, stacklevel=2,
        )
        return _alignment_order(vectors, labels, start_offset)
    return [vector_index for city in order for vector_index in indices_by_city[city]]


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
    dr = abs(step[0]) * spacing
    dc = abs(step[1]) * spacing
    movement_shape = config.get("movement_shape", "trapezoidal")
    if movement_shape == "Cubic_Spline":
        return math.sqrt(6 * dr / acceleration) + math.sqrt(6 * dc / acceleration)
    if movement_shape == "trapezoidal":
        return _profile_seconds(math.hypot(dr, dc), velocity, acceleration)
    raise ValueError(f"Unsupported movement_shape: {movement_shape!r}.")


def _motion_parameters(config):
    return (
        config["lattice_spacing"].to("meter").magnitude / 2,
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


@dataclass(frozen=True)
class _SideTour:
    load_step: tuple[int, int]
    visit_offsets: tuple[tuple[int, int], ...]
    routes: tuple[tuple[tuple[int, int], ...], ...]
    return_steps: tuple[tuple[int, int], ...]
    motion_seconds: float


def _optimize_visit_sides(vectors, config, *, start_offset=None):
    """Minimum parity-routed motion for a FIXED sequence of home-relative targets.

    Keep four endpoint states per visit; compare 4 x 4 transitions between
    adjacent visits, plus four loading/unloading directions at the boundaries.
    This is O(16*n) routing comparisons and O(n) storage, not a new subset TSP.
    All homes are even-even, so routing the shared offset has the same parity
    geometry as routing any loaded atom. Transfers are one cardinal step.
    A non-None start_offset means the load is already in the AOD: start there
    without another transfer, but still price the eventual return to home.
    """
    if not vectors:
        return _SideTour((0, 0), (), (), (), 0.0)
    if any(r % 2 or c % 2 for r, c in vectors):
        raise ValueError("Side optimization requires even-even home-relative targets.")
    if start_offset is not None and not _is_valid_aod_position(start_offset):
        raise ValueError("A reused load must start at an AOD offset.")
    points = [tuple((r + dr, c + dc) for dr, dc in _CARDINAL) for r, c in vectors]

    def edge(start, end):
        return _route(start, end, config)

    transfer_costs = [_step_seconds(step, config) for step in _CARDINAL]
    costs, starts = [], []
    for end in points[0]:
        if start_offset is None:
            cost, start = min((transfer_costs[s] + edge(offset, end)[0], s)
                              for s, offset in enumerate(_CARDINAL))
        else:
            cost, start = edge(start_offset, end)[0], None
        costs.append(cost)
        starts.append(start)
    parents = []
    for previous, current in zip(points, points[1:]):
        next_costs, predecessors = [], []
        for end in current:
            cost, predecessor = min((costs[s] + edge(start, end)[0], s)
                                    for s, start in enumerate(previous))
            next_costs.append(cost)
            predecessors.append(predecessor)
        parents.append(predecessors)
        costs = next_costs
    total, last, exit_side = min(
        (costs[s] + edge(start, end)[0] + transfer_costs[e], s, e)
        for s, start in enumerate(points[-1]) for e, end in enumerate(_CARDINAL)
    )
    chosen = [last]
    for predecessors in reversed(parents):
        chosen.append(predecessors[chosen[-1]])
    chosen.reverse()
    load_step = _CARDINAL[starts[chosen[0]]] if start_offset is None else (0, 0)
    initial = load_step if start_offset is None else start_offset
    offsets = tuple(options[side] for options, side in zip(points, chosen))
    routes = tuple(edge(start, end)[1] for start, end in zip((initial, *offsets[:-1]), offsets))
    unload_offset = _CARDINAL[exit_side]
    return_steps = edge(offsets[-1], unload_offset)[1] + (tuple(-v for v in unload_offset),)
    result = _SideTour(load_step, offsets, routes, return_steps, total)
    return result


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


def _gate_parts(line):
    for statement in line.split(";"):
        statement = statement.strip()
        if statement:
            ids = tuple(map(int, re.findall(r"q\[(\d+)\]", statement)))
            pulse = statement.split(" q[", 1)[0]
            yield statement + ";", pulse, ids


def _blockade_sq(config):
    """Squared RYDBERG_RADIUS in grid steps; one step is lattice_spacing / 2."""
    step = (config.get("_motion_parameters") or _motion_parameters(config))[0]
    blockade_sq = (RYDBERG_RADIUS / step) ** 2
    if blockade_sq <= 1 + 1e-9:
        raise ValueError(
            "lattice_spacing / 2 must be smaller than the Rydberg radius "
            f"({RYDBERG_RADIUS * 1e6:g} micrometers) so gate partners interact."
        )
    return blockade_sq


def _rydberg_conflict(sites, others, blockade_sq):
    """True when an atom of one gate is strictly inside the radius of another's."""
    limit = blockade_sq * (1 - 1e-9)
    return any(
        (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 < limit for a in sites for b in others
    )


def _pulse_coloring(sites, blockade_sq):
    """Split gates into conflict-free pulses with DSATUR; returns index lists.

    Polynomial time and exact whenever two pulses suffice (DSATUR colors
    bipartite graphs optimally); a heuristic for denser conflict graphs.
    """
    n = len(sites)
    neighbors = [
        {j for j in range(n) if j != i and _rydberg_conflict(sites[i], sites[j], blockade_sq)}
        for i in range(n)
    ]
    color = {}
    while len(color) < n:
        i = max(
            (i for i in range(n) if i not in color),
            key=lambda i: (len({color[j] for j in neighbors[i] if j in color}), len(neighbors[i]), -i),
        )
        taken = {color[j] for j in neighbors[i] if j in color}
        color[i] = next(c for c in range(n) if c not in taken)
    return [[i for i in range(n) if color[i] == c] for c in range(max(color.values()) + 1)]


def _append_gate(events, node, position, config, floor=0):
    """Emit a gate into the trailing pulse window using the fewest steps found.

    Consecutive gate events after ``floor`` with no movement between them form
    the window, so every atom is still at ``position(q)``. A gate joins a step
    with the same pulse when it shares no qubit with that step or any later
    one, and, for two-qubit pulses, no atom lies strictly inside RYDBERG_RADIUS
    of an atom of another gate in the step. When it fits no step, the window's
    pulses are recolored together with it, and kept if that saves a timestep.
    """
    line = format_node_line(node)
    _, pulse, ids = next(_gate_parts(line))
    sites = [position(q) for q in ids]
    blockade_sq = _blockade_sq(config) if len(ids) == 2 else None
    used, target, start = set(ids), None, len(events)
    for k in range(len(events) - 1, floor - 1, -1):
        if events[k][0] != "gate":
            break
        parts = list(_gate_parts(events[k][1]))
        if any(p != pulse for _, p, _ in parts):
            break
        step_ids = [q for _, _, qs in parts for q in qs]
        if used & set(step_ids):
            break
        used.update(step_ids)
        start = k
        if blockade_sq is None or not _rydberg_conflict(
            sites, [position(q) for q in step_ids], blockade_sq
        ):
            target = k
    if target is not None:
        events[target] = ("gate", events[target][1] + " " + line)
        return
    if blockade_sq is not None and start < len(events):
        # Window steps share no qubits, so their gates may be regrouped freely.
        gates = [(statement, qs) for event in events[start:] for statement, _, qs in _gate_parts(event[1])]
        gates.append((line, ids))
        steps = _pulse_coloring([[position(q) for q in qs] for _, qs in gates], blockade_sq)
        if len(steps) <= len(events) - start:
            events[start:] = [("gate", " ".join(gates[i][0] for i in step)) for step in steps]
            return
    events.append(("gate", line))


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
        if first[0] != "gate":
            key = _move_batch_key(first)
            used = {first[1]}
            while i < len(event_list):
                nxt = event_list[i]
                if nxt[0] == "gate" or _move_batch_key(nxt) != key or nxt[1] in used:
                    break
                batch.append(nxt)
                used.add(nxt[1])
                i += 1
        yield batch


def schedule_duration(events, config):
    """One common duration model for optimized, reset, and baseline schedules."""
    seconds = 0.0
    transfers = 0
    one_pulses = two_pulses = measurement_batches = 0
    for batch in _event_batches(events):
        first = batch[0]
        if first[0] == "gate":
            pulses = {(pulse, len(ids)) for _, pulse, ids in _gate_parts(first[1])}
            for pulse, arity in pulses:
                if pulse == "measure":
                    measurement_batches += 1
                elif arity == 1:
                    one_pulses += 1
                elif arity == 2:
                    two_pulses += 1
                else:
                    raise ValueError("Only one- and two-qubit pulses are supported.")
        else:
            seconds += max(_step_seconds(_movement_vector(e[2], e[3]), config) for e in batch)
            # A step whose atoms are all held changes no traps.
            transfers += int(any(
                e[0] != "hold" and not (_is_valid_aod_position(e[2]) and _is_valid_aod_position(e[3]))
                for e in batch
            ))
    return (
        seconds * config["t_switch"].to("seconds").units
        + transfers * config["transfer_SLM_AOD"]
        + one_pulses * (config["average_single_gate_time"] + config["t_switch"])
        + two_pulses * (config["average_two_gate_time"] + config["t_switch"])
        + measurement_batches * (
            config.get("measurement_time", 0 * config["t_switch"]) + config["t_switch"]
        )
    )


@dataclass
class _Plan:
    positions: dict[int, tuple[int, int]]
    ids: list[int]
    events: list[ScheduleEvent]
    reused_groups: int = 0
    # Gates never pack into pulse steps before this event index.
    pulse_floor: int = 0

    def copy(self):
        return _Plan(
            dict(self.positions), list(self.ids), list(self.events),
            self.reused_groups, self.pulse_floor,
        )


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


def _preferred_mover(pair, config):
    """Return the ancilla operand when exactly one operand is an ancilla."""
    if not config.get("prefer_moving_ancillas", False):
        return None
    ancillas = config.get("_ancilla_qubits", frozenset())
    preferred = [q for q in pair if q in ancillas]
    return preferred[0] if len(preferred) == 1 else None


def _reuse_movers(nodes, previous_ids, config):
    """Single legality predicate used for both reuse selection and execution."""
    previous = set(previous_ids)
    movers = []
    for node in nodes:
        operands = set(op_node_signature(node)[2])
        loaded = operands & previous
        if len(operands) != 2 or len(loaded) != 1:
            return None
        mover = next(iter(loaded))
        preferred = _preferred_mover(operands, config)
        if preferred is not None and mover != preferred:
            return None
        movers.append(mover)
    return movers


def _build_groups(nodes, positions, config, next_use=None):
    """Group only in the state that will actually be used after a reset.

    Look-ahead: unless an ancilla is preferred, the operand needed again
    soonest (``next_use``: qubit -> next layer index) moves, so its load can be
    retained; ties keep operand order. Groups whose movers are needed soonest
    run last, leaving that load active for the next layer.
    """
    next_use = next_use or {}

    def soonest(qubits):
        return min((next_use.get(q, math.inf) for q in qubits), default=math.inf)

    groups = []
    spans = _max_grid_spans(config["max_dimension"])
    for node in nodes:
        pair = op_node_signature(node)[2]
        if len(pair) != 2:
            raise ValueError("A two-qubit layer must contain only two-qubit gates.")
        choices = [(q, _movement_vector(positions[q], positions[pair[1 - i]])) for i, q in enumerate(pair)]
        preferred = _preferred_mover(pair, config)
        if preferred is not None:
            choices = [choice for choice in choices if choice[0] == preferred]
        else:
            choices.sort(key=lambda choice: soonest([choice[0]]))
        placed = False
        # Gates share a load whenever the AOD span allows; a [1, 1] AOD has
        # zero span, so it moves one atom per load.
        for group, movers, vectors in groups:
            fits = [
                (q, vector) for q, vector in choices
                if _fits_same_aod([positions[m] for m in movers], positions[q], *spans)
                and _vector_alignment_score(vectors, vector) >= config.get("alignment_conc", 0)
            ]
            if fits:
                q, vector = max(fits, key=lambda item: (
                    -soonest([item[0]]), _vector_alignment_score(vectors, item[1]),
                ))
                group.append(node)
                movers.append(q)
                vectors.append(vector)
                placed = True
                break
        if not placed:
            q, vector = choices[0]
            groups.append(([node], [q], [vector]))

    #now we order the movers based on displacement
    for group, movers, vectors in groups:
        ordered = sorted(
            zip(group, movers, vectors),
            key=lambda item: (item[2][0], item[2][1], item[1]),
        )
        group[:] = [item[0] for item in ordered]
        movers[:] = [item[1] for item in ordered]
        vectors[:] = [item[2] for item in ordered]

    # Disjoint groups commute; run the soonest-needed load last (stable on ties).
    groups.sort(key=lambda group: -soonest(group[1]))
    return [(group, movers) for group, movers, _ in groups]


def _execute_group(plan, nodes, movers, homes, config, reuse=False):
    if reuse:
        if _reuse_movers(nodes, plan.ids, config) != movers:
            raise ValueError("Cannot reuse an AOD load containing both interacting operands.")
    elif plan.ids:
        raise ValueError("A new group requires the preceding load to be returned.")
    offsets = {
        _movement_vector(homes[q], plan.positions[q])
        for q in set(movers) | set(plan.ids)
    }
    if len(offsets) != 1:
        raise ValueError("Group movers must share one rigid displacement from home.")
    start_offset = offsets.pop()
    home_targets = []
    for node, mover in zip(nodes, movers):
        pair = op_node_signature(node)[2]
        partner = pair[1] if pair[0] == mover else pair[0]
        home_targets.append(_movement_vector(homes[mover], plan.positions[partner]))
    # Order home-relative targets with the side-priced TSP tour.
    order = _order_group_displacements(home_targets, config, start_offset=start_offset)

    nodes = [nodes[i] for i in order]
    ids = [movers[i] for i in order]
    ids.extend(q for q in plan.ids if q not in ids)
    positions = [plan.positions[q] for q in ids]
    index = {q: i for i, q in enumerate(ids)}

    def position(q):
        return positions[index[q]] if q in index else plan.positions[q]

    moves = []
    # Independent gates may be routed in any order, regardless of gate type
    # or CZ preprocessing. Shared-wire order remains fixed by the layers.
    tour = _optimize_visit_sides(
        [home_targets[i] for i in order], config,
        start_offset=start_offset if reuse else None,
    )
    _translate(ids, positions, tour.load_step, plan.events, moves)
    loaded = set(ids)
    for i, (node, steps) in enumerate(zip(nodes, tour.routes)):
        for step in steps:
            _translate(ids, positions, step, plan.events, moves)
        pair = op_node_signature(node)[2]
        if set(pair) & loaded != {ids[i]}:
            raise ValueError("Each interaction must have exactly one loaded operand.")
        partner = pair[1] if pair[0] == ids[i] else pair[0]
        target = plan.positions[partner]
        if _is_valid_aod_position(target) or sum(abs(v) for v in _movement_vector(positions[i], target)) != 1:
            raise ValueError("Side-planned interaction must be adjacent to an SLM partner.")
        _append_gate(plan.events, node, position, config, plan.pulse_floor)
    # The return is a cost-to-go estimate. Keep the load active so the next
    # layer can reuse it; _return_plan executes the return when needed.
    plan.positions.update(zip(ids, positions))
    plan.ids = ids
    plan.reused_groups += int(reuse)


def _one_atom_aod(config):
    """The same configuration with a [1, 1] AOD: one atom per load."""
    return dict(config, max_dimension=[1, 1])


def _adopt(plan, other):
    """Replace ``plan``'s state with ``other``'s, in place."""
    plan.positions, plan.ids, plan.events = other.positions, other.ids, other.events
    plan.reused_groups, plan.pulse_floor = other.reused_groups, other.pulse_floor


def _reset_layer(plan, nodes, homes, config, next_use=None):
    """Execute ``nodes`` from home with the cheaper of grouped and one-atom loads.

    The one-atom-per-load partition (a [1, 1] AOD) is also tried, so a layer's
    grouped execution is never costlier than moving atoms individually from
    the same state.
    """
    if not nodes:
        return
    _return_plan(plan, homes, config)
    # All grouping decisions now see the state after the proposed return.
    partitions = [_build_groups(nodes, plan.positions, config, next_use)]
    if len(nodes) > 1:
        single = _build_groups(nodes, plan.positions, _one_atom_aod(config), next_use)
        if single != partitions[0]:
            partitions.append(single)
    best = best_cost = None
    for groups in partitions:
        trial = plan.copy()
        for group, movers in groups:
            _return_plan(trial, homes, config)
            _execute_group(trial, group, movers, homes, config)
        cost = _closed_cost(trial, homes, config) if len(partitions) > 1 else 0
        if best is None or cost < best_cost:
            best, best_cost = trial, cost
    _adopt(plan, best)


def _closed_cost(plan, homes, config):
    ending = _Plan(dict(plan.positions), list(plan.ids), [])
    _return_plan(ending, homes, config)
    return schedule_duration(plan.events + ending.events, config)


def _plan_layer(nodes, initial, homes, config, stage, next_use, allow_reuse=True):
    reset = initial.copy()
    _reset_layer(reset, nodes, homes, config, next_use)
    if not allow_reuse or not initial.ids or not nodes:
        return reset
    # Gates in a DAG layer are disjoint. Only gates with one loaded operand
    # can precede a reset; AOD-AOD gates remain in the reset portion.
    eligible = [n for n in nodes if _reuse_movers([n], initial.ids, config) is not None]
    # All eligible gates may ride the retained load; a single one is also
    # tried, so grouping is never costlier than individual moves from here.
    options = [eligible[:1]]
    if len(eligible) > 1:
        options.insert(0, eligible)
    # Compare equal endpoints (all atoms at home). The closure is a cost-to-go
    # estimate, not prematurely committed: the next layer can still reuse.
    best, best_cost = reset, _closed_cost(reset, homes, config)
    for chosen in options:
        movers = _reuse_movers(chosen, initial.ids, config)
        if not movers or not _reuse_group_allowed(
            initial.ids, movers, config.get("T_reuse", 0), stage, next_use
        ):
            continue
        reuse = initial.copy()
        _execute_group(reuse, chosen, movers, homes, config, reuse=True)
        selected = {id(n) for n in chosen}
        _reset_layer(reuse, [n for n in nodes if id(n) not in selected], homes, config, next_use)
        cost = _closed_cost(reuse, homes, config)
        if cost < best_cost:
            best, best_cost = reuse, cost
    return best


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
    blockade_sq = _blockade_sq(config)
    for batch in _event_batches(events):
        if batch[0][0] == "gate":
            used = set()
            pulsed = []
            for statement, _, ids in _gate_parts(batch[0][1]):
                if not ids or not set(ids) <= positions.keys() or used & set(ids):
                    raise ValueError("Invalid operands or overlapping simultaneous gates.")
                used.update(ids)
                if len(ids) == 2:
                    if len(set(ids) & active) != 1:
                        raise ValueError("A two-qubit gate requires one AOD and one SLM atom.")
                    preferred = _preferred_mover(ids, config)
                    if preferred is not None and preferred not in active:
                        raise ValueError("A data-ancilla gate must move its ancilla operand.")
                    if sum(abs(v) for v in _movement_vector(positions[ids[0]], positions[ids[1]])) != 1:
                        raise ValueError("Two-qubit gate separation is not one grid step.")
                    sites = [positions[q] for q in ids]
                    if any(_rydberg_conflict(sites, other, blockade_sq) for other in pulsed):
                        raise ValueError("Simultaneous two-qubit gates are within the Rydberg radius.")
                    pulsed.append(sites)
                elif len(ids) != 1:
                    raise ValueError("Only one- and two-qubit gates are supported.")
                for q in ids:
                    traces[q].append(statement)
            continue
        moved = {e[1] for e in batch}
        # Held atoms stay in the AOD while a transfer step carries them over
        # their SLM home; only the other atoms of the step change traps.
        held = {e[1] for e in batch if e[0] == "hold"}
        if any(e[0] not in {"move", "hold"} for e in batch):
            raise ValueError("Unknown movement event kind.")
        first = batch[0]
        loading = not _is_valid_aod_position(first[2])
        unloading = not _is_valid_aod_position(first[3])
        if loading and unloading:
            raise ValueError("Direct SLM-to-SLM movement is forbidden.")
        if held and not (loading or unloading):
            raise ValueError("Held atoms may only ride a load or unload step.")
        if not held <= active:
            raise ValueError("Only atoms already in the AOD can be held.")
        if loading:
            if active != held:
                raise ValueError("Load attempted before returning the existing AOD load.")
            span_positions = [e[2] for e in batch]
            if not all(_fits_same_aod(span_positions[:i], p, *_max_grid_spans(config["max_dimension"]))
                       for i, p in enumerate(span_positions)):
                raise ValueError("AOD load exceeds the configured span.")
            active = moved.copy()
        elif moved != active:
            raise ValueError("The entire AOD load must move or unload together.")
        for kind, q, start, end in batch:
            if q not in positions or positions[q] != start:
                raise ValueError("Movement continuity violation.")
            if loading or unloading:
                if sum(abs(v) for v in _movement_vector(start, end)) != 1:
                    raise ValueError("Transfers must be one cardinal grid step.")
                if loading and _is_valid_aod_position(start):
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
            active = held
    if require_home and (active or positions != homes):
        raise ValueError("The completed circuit must return every atom to its home.")
    if reference_nodes is not None:
        _validate_barrier_order(events, reference_nodes)
        expected = defaultdict(list)
        commuting, fixed = set(), set()
        for node in reference_nodes:
            if node.op.name in {"swap", "barrier"}:
                continue
            if len(node.qargs) not in (1, 2):
                continue
            line = format_node_line(node)
            (commuting if _is_reorderable_cz(node) else fixed).add(line)
            for q in op_node_signature(node)[2]:
                expected[q].append(line)
        if not _cz_commutation_enabled(config):
            commuting = set()
        commuting -= fixed
        if _cz_run_traces(traces, commuting) != _cz_run_traces(expected, commuting):
            raise ValueError("Gate identities, multiplicities, or per-qubit order changed.")
    return positions


def _cz_commutation_enabled(config):
    """CZ preprocessing opts in to reordering commuting unconditional CZs."""
    return bool(config.get("reorder_cz", False) or config.get("reorder_cz_by_displacement", False))


def _cz_run_traces(traces, commuting):
    """Per-wire traces where each maximal run of ``commuting`` CZs is a multiset.

    Unconditional CZs are diagonal and commute; any other operation on the
    wire ends a run, as in ``dag_helper._validate_cz_reordering``.
    """
    result = {}
    for q, trace in traces.items():
        canonical, run = [], []
        for statement in trace:
            if statement in commuting:
                run.append(statement)
            else:
                canonical.extend((tuple(sorted(run)), statement))
                run = []
        canonical.append(tuple(sorted(run)))
        result[q] = canonical
    return result


def _validate_barrier_order(events, reference_nodes):
    """Require every emitted operation to remain in its QASM barrier epoch."""
    epochs = [defaultdict(int)]
    for node in reference_nodes:
        if node.op.name == "swap":
            continue
        if node.op.name == "barrier":
            epochs.append(defaultdict(int))
        elif len(node.qargs) in (1, 2):
            epochs[-1][format_node_line(node)] += 1

    epoch = 0
    for event in events:
        if event[0] != "gate":
            continue
        for statement, _, _ in _gate_parts(event[1]):
            while epoch < len(epochs) - 1 and not any(epochs[epoch].values()):
                epoch += 1
            if epochs[epoch].get(statement, 0):
                epochs[epoch][statement] -= 1
            elif any(later.get(statement, 0) for later in epochs[epoch + 1:]):
                raise ValueError("An operation crossed a QASM barrier.")


def _optimize_retained_loads(events, layers, homes, config, singles=None):
    """Jointly route actual load-to-unload intervals without changing retention.

    The ordinary planner has already selected legal loads under T_reuse. Here
    only their internal routes change. Layer boundaries and intervening 1Q
    contexts remain ordered, and no atom is added to or removed from a load.
    With CZ preprocessing enabled, consecutive CZ-only layers with nothing
    between them (``singles`` empty, no barrier-emptied layer) commute and form
    one routing stage, so their gates may be visited in any order.
    """
    merge = singles is not None and _cz_commutation_enabled(config)
    cz_layers = [bool(nodes) and all(map(_is_reorderable_cz, nodes)) for nodes in layers]

    def same_stage(previous, layer):
        return merge and previous is not None and previous < layer and cz_layers[previous] and all(
            cz_layers[k] and not singles[k] for k in range(previous + 1, layer + 1)
        )

    occurrences = defaultdict(deque)
    for stage, nodes in enumerate(layers):
        for node in nodes:
            occurrences[format_node_line(node)].append((stage, node))

    def reroute(interval, ids):
        visits, targets, labels, contexts = [], [], [], {}
        pending = []
        stage, last_layer = -1, None
        original_layers = set()
        loaded = set(ids)
        for event in interval:
            if event[0] != "gate":
                continue
            parts = list(_gate_parts(event[1]))
            if all(len(pair) == 1 for _, _, pair in parts):
                pending.append(event)
                continue
            for statement, _, pair in parts:
                if len(pair) != 2 or not occurrences[statement]:
                    raise ValueError("Retained-load planning lost a gate occurrence.")
                layer, node = occurrences[statement].popleft()
                if pending or (layer != last_layer and not same_stage(last_layer, layer)):
                    stage += 1
                    contexts[stage] = pending
                    pending = []
                last_layer = layer
                movers = set(pair) & loaded
                if len(movers) != 1:
                    raise ValueError("Retained interaction must have exactly one loaded operand.")
                mover = next(iter(movers))
                partner = pair[1] if pair[0] == mover else pair[0]
                visits.append((node, mover, partner))
                targets.append(_movement_vector(homes[mover], homes[partner]))
                labels.append(stage)
                original_layers.add(layer)
        if len(original_layers) < 2:
            return interval

        order = _order_group_displacements(targets, config, layer_ids=labels)
        if sorted(order) != list(range(len(visits))) or [labels[i] for i in order] != sorted(labels):
            raise ValueError("Retained-load tour changed gate occurrences or layer order.")
        plan = _Plan(dict(homes), list(ids), [])

        def translate(step):
            positions = [plan.positions[q] for q in ids]
            _translate(ids, positions, step, plan.events, [])
            plan.positions.update(zip(ids, positions))

        tour = _optimize_visit_sides([targets[i] for i in order], config)
        translate(tour.load_step)
        previous_stage = None
        for position, index in enumerate(order):
            if labels[index] != previous_stage:
                plan.events.extend(contexts[labels[index]])
                previous_stage = labels[index]
                plan.pulse_floor = len(plan.events)
            node, mover, partner = visits[index]
            for step in tour.routes[position]:
                translate(step)
            if sum(abs(v) for v in _movement_vector(plan.positions[mover], homes[partner])) != 1:
                raise ValueError("Retained-load visit is not adjacent to its partner.")
            _append_gate(plan.events, node, plan.positions.__getitem__, config, plan.pulse_floor)
        plan.events.extend(pending)
        for step in tour.return_steps:
            translate(step)
        if plan.positions != homes:
            raise ValueError("Retained-load tour did not restore original homes.")
        return plan.events

    result, interval, ids = [], [], []
    for batch in _event_batches(events):
        first = batch[0]
        if first[0] == "move" and not _is_valid_aod_position(first[2]):
            if ids:
                raise ValueError("Cannot start a new load before unloading the previous load.")
            ids = [event[1] for event in batch]
            if any(start != homes[q] for _, q, start, _ in batch):
                raise ValueError("A retained-load tour must start at the original homes.")
        if ids:
            interval.extend(batch)
            if first[0] == "move" and not _is_valid_aod_position(first[3]):
                result.extend(reroute(interval, ids))
                interval, ids = [], []
        else:
            result.extend(batch)
    if ids or any(occurrences.values()):
        raise ValueError("Retained-load planning did not consume the complete circuit.")
    return result


def _hold_reloaded_atoms(events):
    """Keep atoms in the AOD through an unload that is followed by their reload.

    The planner releases a whole load before picking up the next one. An atom
    unloaded at home and loaded again by the very next transfer need not
    leave the AOD: it rides both rigid transfer steps as a "hold", passing
    over its home site, and only atoms that actually change traps are loaded
    or unloaded. Atoms measured or reset in between are released as before.
    """
    batches = [list(batch) for batch in _event_batches(events)]

    def transfer(batch, index):
        return batch[0][0] != "gate" and not _is_valid_aod_position(batch[0][index])

    for i, unload in enumerate(batches):
        if not transfer(unload, 3):
            continue
        j = next((k for k in range(i + 1, len(batches)) if batches[k][0][0] != "gate"), None)
        if j is None or not transfer(batches[j], 2):
            continue
        released = {
            q for batch in batches[i + 1:j]
            for _, pulse, ids in _gate_parts(batch[0][1])
            if pulse.startswith(("measure", "reset")) for q in ids
        }
        kept = ({e[1] for e in unload} & {e[1] for e in batches[j]}) - released
        for batch in (unload, batches[j]):
            batch[:] = [("hold", *e[1:]) if e[1] in kept else e for e in batch]
    return [event for batch in batches for event in batch]


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
            singles[i], config["average_single_gate_time"], config["t_switch"],
            config.get("measurement_time"),
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
        singles[-1], config["average_single_gate_time"], config["t_switch"],
        config.get("measurement_time"),
    )
    plan.events.extend(("gate", line) for line in lines)
    _return_plan(plan, homes, config)
    plan.events = _optimize_retained_loads(plan.events, layers, homes, config, singles)
    plan.events = _hold_reloaded_atoms(plan.events)
    return plan


@dataclass
class CircuitSchedule:
    events: list[ScheduleEvent]
    duration: Any
    selected: str
    candidate_times: dict[str, Any]
    rejected_candidates: dict[str, str]
    reused_groups: int


def schedule_circuit(layers, singles, qubits, config, reference_nodes=None):
    """Build, validate, and return the cheaper layered schedule.

    ``reuse`` is the layered planner with the configured AOD size and
    ``T_reuse``; gates share loads whenever the AOD span allows. Unless the
    AOD is already [1, 1], the planner also runs with a [1, 1] AOD
    (``individual``), so a run never does worse than moving one atom at a
    time. Both are validated and timed with the common model; the cheaper is
    returned (``reuse`` on ties).
    The side-priced TSP tour and side DP apply to ordinary groups regardless
    of CZ reordering. Planning or validation failures propagate.
    """
    if "_side_ordering" in config:
        raise ValueError(_SIDE_ORDERING_REMOVED)
    if "parallel" in config:
        raise ValueError(_PARALLEL_REMOVED)
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
    config["_ancilla_qubits"] = frozenset(
        q for node in reference_nodes if node.op.name == "reset"
        for q in op_node_signature(node)[2]
    )
    builders = {"reuse": lambda: _layered_candidate(layers, singles, homes, config, True)}
    if list(config["max_dimension"]) != [1, 1]:
        individual = _one_atom_aod(config)
        builders["individual"] = lambda: _layered_candidate(layers, singles, homes, individual, True)
    plans, times = {}, {}
    for name, build in builders.items():
        plans[name] = build()
        validate_schedule(plans[name].events, homes, config, reference_nodes)
        times[name] = schedule_duration(plans[name].events, config)
    selected = min(times, key=times.get)
    plan = plans[selected]
    return CircuitSchedule(
        plan.events, times[selected], selected, times, {}, plan.reused_groups,
    )
