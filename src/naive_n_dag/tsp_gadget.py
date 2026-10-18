"""Exact side-aware displacement tours via a Noon-Bean + Jonker-Volgenant gadget.

The relaxed tour in ``dynamics`` prices every edge with the cheapest pair of
interaction sides, so it may arrive at a city on one side and leave from
another. Here each city is a cluster of its four side sites (a generalized
TSP), Noon-Bean turns the clusters into an asymmetric TSP, and the
Jonker-Volgenant in/out split turns that into the symmetric TSP Concorde
solves. An optimal tour of the gadget is an optimal visit order *and* side
choice under parity-routed motion times. TSP_GADGET.md has the construction
and the proofs; this module only builds the matrix and decodes tours.

Pure graph code: routing and timing come in as callables, so nothing here
depends on the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass

INT_MAX = 2**31 - 1
# Largest edge weight handed to Concorde: tour lengths are doubles there, but
# Lin-Kernighan sums move gains (at most 50 edges deep) in an int. Matches
# dynamics._CONCORDE_EDGE_CAP.
EDGE_CAP = INT_MAX // 256
_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
# Integer units given to the tour upper bound (K is just above it). Large
# enough that rounding is negligible; 4K stays under EDGE_CAP.
_UPPER_BOUND_UNITS = 10**6
# Below this many integer units per route edge the rounding swamps the motion
# times; such instances are left to the relaxed tour.
_MIN_UNITS_PER_EDGE = 10


class GadgetError(RuntimeError):
    """The gadget cannot be built, or a solver tour does not decode faithfully."""


@dataclass(frozen=True)
class GadgetGraph:
    """Symmetric integer TSP instance for one AOD load.

    ATSP node ``a`` becomes symmetric nodes ``2a`` (in) and ``2a + 1`` (out).
    Cluster 0 is the load terminal, clusters 1..C the cities in input order,
    and cluster C + 1 the unload terminal. ``clusters`` lists each cluster's
    ATSP nodes in Noon-Bean cycle order. Edges absent from the gadget weigh
    ``missing`` = 4K, which already makes any tour using one at least K worse
    than a faithful tour (TSP_GADGET.md, "Absent edges").
    """

    weights: tuple[tuple[int, ...], ...]
    missing: int
    K: int
    clusters: tuple[tuple[int, ...], ...]
    cluster_layers: tuple[int, ...]
    node_cluster: tuple[int, ...]
    node_site: tuple[tuple[int, int], ...]


def build_gadget(cities, start_offset, route, transfer, upper_bound):
    """Build the gadget for labelled cities ``(layer, (row, col))``.

    ``cities`` are distinct, with even-even home-relative targets and layers
    normalized to 0..L-1. ``start_offset`` is (0, 0) for a load from home
    (four transfer directions) or the AOD offset of a retained load.
    ``route(a, b)`` is the parity-routed motion seconds between two sites and
    ``transfer(step)`` the seconds of one cardinal SLM<->AOD transfer.
    ``upper_bound`` is the exact motion seconds of any feasible tour (the
    relaxed tour); K must exceed the optimum, and this bounds it.
    """
    if not cities:
        raise GadgetError("The gadget needs at least one city.")
    if any(r % 2 or c % 2 for _, (r, c) in cities):
        raise GadgetError("Gadget cities must be even-even home-relative targets.")
    if start_offset != (0, 0) and start_offset[0] % 2 == 0 and start_offset[1] % 2 == 0:
        raise GadgetError("A retained load must start at an AOD site.")
    if not upper_bound > 0:
        raise GadgetError("The tour upper bound must be positive.")
    layer_count = max(layer for layer, _ in cities) + 1
    if sorted({layer for layer, _ in cities}) != list(range(layer_count)):
        raise GadgetError("City layers must be normalized to 0..L-1.")

    # (site, depart seconds, arrive seconds) per ATSP node, grouped by cluster.
    if start_offset == (0, 0):
        load = [(step, transfer(step), 0.0) for step in _CARDINAL]
    else:
        load = [(start_offset, 0.0, 0.0)]
    spec = [load]
    spec += [[((r + dr, c + dc), 0.0, 0.0) for dr, dc in _CARDINAL] for _, (r, c) in cities]
    spec.append([(step, 0.0, transfer(step)) for step in _CARDINAL])
    layers = (-1, *(layer for layer, _ in cities), layer_count)
    unload = len(spec) - 1

    clusters, node_cluster, node_site, depart, arrive = [], [], [], [], []
    for index, nodes in enumerate(spec):
        clusters.append(tuple(range(len(node_cluster), len(node_cluster) + len(nodes))))
        for site, leave, enter in nodes:
            node_cluster.append(index)
            node_site.append(site)
            depart.append(leave)
            arrive.append(enter)
    count = len(node_cluster)
    successor = {}
    for nodes in clusters:
        for i, node in enumerate(nodes):
            successor[node] = nodes[(i + 1) % len(nodes)]

    def allowed(x, y):
        """Directed cluster arcs: same or next layer, plus the unload->load wrap."""
        if x == y:
            return False
        if x == unload:
            return y == 0
        return y != 0 and layers[y] - layers[x] in (0, 1)

    def seconds(a, b):
        """Motion of one ATSP arc a->b: leave a's site, route, enter b's site."""
        if node_cluster[a] == unload:
            return 0.0
        return depart[a] + route(node_site[a], node_site[b]) + arrive[b]

    # Integer budget (TSP_GADGET.md §6). Every weight is at most 4K <= EDGE_CAP.
    # Concorde's exact bounds use fixed-point arithmetic with about an int32
    # range ("OVERFLOW in CCbigguy_addmult" aborts the process), so a faithful
    # tour, (W+1)*K, stays at most INT_MAX/8, and even n absent edges fit.
    n = 2 * count
    total = sum(len(nodes) + 1 for nodes in clusters)
    k_max = min(EDGE_CAP // 4, INT_MAX // (8 * (total + 1)), INT_MAX // (4 * n))
    route_edges = len(cities) + 1
    units = min(_UPPER_BOUND_UNITS, k_max - route_edges - 1)
    if units < _MIN_UNITS_PER_EDGE * route_edges:
        raise GadgetError(
            f"{len(cities)} cities leave too little int32 resolution for the gadget."
        )
    scale = units / upper_bound
    # Rounding moves each of the route_edges route weights of a faithful tour
    # by at most 1/2, so K stays above the integer optimum.
    K = int(upper_bound * scale) + route_edges + 1
    missing = 4 * K

    rows = [[missing] * n for _ in range(n)]

    def connect(i, j, weight):
        rows[i][j] = rows[j][i] = weight

    for a in range(count):
        connect(2 * a, 2 * a + 1, 0)  # Jonker-Volgenant tie
        x = node_cluster[a]
        if len(clusters[x]) > 1:
            connect(2 * a + 1, 2 * successor[a], K)  # Noon-Bean cycle arc
        for b in range(count):
            y = node_cluster[b]
            if allowed(x, y):
                # Noon-Bean shift: leaving a is priced as leaving successor(a).
                cost = round(seconds(successor[a], b) * scale)
                connect(2 * a + 1, 2 * b, cost + 2 * K)
    return GadgetGraph(
        tuple(map(tuple, rows)), missing, K, tuple(clusters), layers,
        tuple(node_cluster), tuple(node_site),
    )


def decode_gadget_tour(tour, graph):
    """Return ``(city order, entry sites)`` after checking the tour is faithful.

    The tour must be a Hamiltonian cycle over present edges that uses every
    in/out tie in one orientation (so it is a directed ATSP tour), visits
    each cluster once as a full forward pass of its Noon-Bean cycle, runs
    from the load terminal to the unload terminal, and never decreases in
    layer. Each city's entry site is its interaction side.
    """
    n = len(graph.weights)
    tour = [int(node) for node in tour]
    if len(tour) != n or set(tour) != set(range(n)):
        raise GadgetError("Solver did not return a Hamiltonian tour of the gadget.")
    if any(graph.weights[a][b] >= graph.missing for a, b in zip(tour, tour[1:] + tour[:1])):
        raise GadgetError("Solver tour uses an edge absent from the gadget.")
    position = {node: i for i, node in enumerate(tour)}
    orientation = set()
    for a in range(n // 2):
        i, o = position[2 * a], position[2 * a + 1]
        if (i + 1) % n == o:
            orientation.add(True)
        elif (o + 1) % n == i:
            orientation.add(False)
        else:
            raise GadgetError("Solver tour splits a Jonker-Volgenant in/out pair.")
    if len(orientation) != 1:
        raise GadgetError("Solver tour traverses in/out pairs in both directions.")
    if orientation == {False}:
        tour.reverse()
    first = next(i for i, node in enumerate(tour) if node % 2 == 0)
    tour = tour[first:] + tour[:first]
    sequence = [node // 2 for node in tour[::2]]

    # Rotate to the load cluster's entry, then split into cluster runs. There
    # are always at least three clusters, so some run of cluster 0 has an entry.
    start = next(
        i for i in range(len(sequence))
        if graph.node_cluster[sequence[i]] == 0 and graph.node_cluster[sequence[i - 1]] != 0
    )
    sequence = sequence[start:] + sequence[:start]
    runs = []
    for node in sequence:
        if runs and graph.node_cluster[runs[-1][-1]] == graph.node_cluster[node]:
            runs[-1].append(node)
        else:
            runs.append([node])
    order = [graph.node_cluster[run[0]] for run in runs]
    if sorted(order) != list(range(len(graph.clusters))):
        raise GadgetError("Solver tour visits a cluster more than once.")
    for run in runs:
        nodes = graph.clusters[graph.node_cluster[run[0]]]
        k = nodes.index(run[0])
        if run != list(nodes[k:] + nodes[:k]):
            raise GadgetError("Solver tour does not pass a cluster's sites in cycle order.")
    if order[0] != 0 or order[-1] != len(graph.clusters) - 1:
        raise GadgetError("Solver tour does not run from loading to unloading.")
    layers = [graph.cluster_layers[x] for x in order]
    if layers != sorted(layers):
        raise GadgetError("Solver tour violated layer order.")
    cities = [x - 1 for x in order[1:-1]]
    sites = [graph.node_site[run[0]] for run in runs[1:-1]]
    return cities, sites
