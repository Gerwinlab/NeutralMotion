"""Deterministic initial placement: ``fill_strategy: "deterministic"``.

It minimises F_time (placement_cost.py), the cost FastSA anneals, on the
same two-qubit DAG, but with no random numbers:

  0. presolve   closed forms for star and path circuits
  1. seeds      interaction-graph orders laid along geometric paths, layouts
                that realise the circuit's translation symmetry, product
                layouts of hypergraph-product codes (optional), row-major,
                and any layouts supplied with the ``seed_layouts`` option
  2. search     best-first multistart over the seeds, each descended by a
                variable neighbourhood descent (swap/relocate moves with
                candidate lists, and layer-guided moves)
  3. screening  optional (``schedule_screen``): the result, the default-pool
                pick and the best seeds go back to ``main``, which schedules
                them and keeps the fastest (see _screen_candidates)

Every search evaluation is counted against ``budget``. Ties are broken by
index, so a run is reproducible.

Layouts here are lists indexed by qubit of ``(row, col)`` SLM *site* indices
in ``0..R-1 x 0..C-1``; grid coordinates are ``(2*row, 2*col)``.
"""

from __future__ import annotations

import collections
import heapq
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dag_helper import extract_index_from_bit
from .grid import Qubit, place_qubit
from .placement_cost import make_cost

Layout = list[tuple[int, int]]

# FastSA's evaluation count on the qLDPC benchmark configs; a cap, not a target.
DEFAULT_BUDGET = 100_103

DEFAULT_OPTIONS: dict[str, Any] = {
    "budget": DEFAULT_BUDGET,
    # Nearest-qubit candidate sizes: each k adds a swap phase with k nearest
    # qubits (and, with layer_moves, a layer phase, which does not use k).
    "k_ladder": [12, 24, 48],
    # Well-spread "distant" swap candidates per qubit (golden-ratio walk).
    "k_disc": 6,
    # Nearest free sites tried when relocating a qubit.
    "free_per_qubit": 4,
    # Layer-guided neighbourhood inside the VND.
    "layer_moves": True,
    # Seeds that realise a discovered translation group Z_L x Z_M.
    "symmetry_seeds": True,
    # Star / path closed forms when every layer holds one gate.
    "closed_forms": True,
    # Torus seeds from every generator pair of the translation group, not the
    # first pair per shape.
    "torus_all_pairs": False,
    # Product seeds for circuits whose interaction graph is a certified
    # Cartesian product (hypergraph-product codes): qubit (u, v) at site
    # (order0(u), order1(v)).
    "product_seeds": False,
    # Screening: hand main the result plus the best distinct seeds by the cost,
    # up to this many; main ranks them by a fast schedule and keeps the
    # fastest full schedule. 0: off (the result is the search's layout).
    "schedule_screen": 0,
    # Candidates main schedules in full after the fast ranking (the search's
    # result and the current-code pick always are).
    "screen_final": 3,
    # Layout JSON files ([row, col] site per qubit, as initial_layout) added
    # to the seed pool as "given:<file>". Empty: every seed is constructed.
    "seed_layouts": [],
    # Diversity veto: skip a seed whose sites differ in fewer than this many
    # qubits from a layout already descended (or a descent's end point).
    # None means max(2, n // 6).
    "min_sep": None,
    # Stop after this many descents without improvement (None: never)...
    "stall": 10,
    # ...but only once this share of the budget has been spent.
    "min_spend_fraction": 0.10,
}

_TOL = 1e-12
_PHI_INV = 0.6180339887498949  # 1 / golden ratio


def validate_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merged and checked ``Deterministic_fill`` options."""
    options = dict(options or {})
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(
            f"Unknown Deterministic_fill option(s): {', '.join(sorted(unknown))}. "
            f"Known: {', '.join(sorted(DEFAULT_OPTIONS))}."
        )
    merged = {**DEFAULT_OPTIONS, **options}

    def positive_int(key, allow_zero=False):
        value = merged[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
            raise ValueError(f"Deterministic_fill.{key} must be a {'nonnegative' if allow_zero else 'positive'} integer.")

    positive_int("budget")
    positive_int("k_disc", allow_zero=True)
    positive_int("free_per_qubit", allow_zero=True)
    positive_int("schedule_screen", allow_zero=True)
    positive_int("screen_final")
    ladder = merged["k_ladder"]
    if not isinstance(ladder, (list, tuple)) or not ladder or not all(
        isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in ladder
    ):
        raise ValueError("Deterministic_fill.k_ladder must be a non-empty list of positive integers.")
    merged["k_ladder"] = list(ladder)
    for key in ("layer_moves", "symmetry_seeds", "closed_forms", "torus_all_pairs", "product_seeds"):
        if not isinstance(merged[key], bool):
            raise ValueError(f"Deterministic_fill.{key} must be true or false.")
    for key in ("min_sep", "stall"):
        if merged[key] is not None:
            positive_int(key)
    seed_layouts = merged["seed_layouts"]
    if not isinstance(seed_layouts, (list, tuple)) or not all(isinstance(p, str) for p in seed_layouts):
        raise ValueError("Deterministic_fill.seed_layouts must be a list of layout file paths.")
    merged["seed_layouts"] = list(seed_layouts)
    fraction = merged["min_spend_fraction"]
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0 <= fraction <= 1:
        raise ValueError("Deterministic_fill.min_spend_fraction must be in [0, 1].")
    return merged


class _Evaluator:
    """Counts objective evaluations against the budget."""

    def __init__(self, objective, budget: int):
        self.objective = objective
        self.budget = budget
        self.count = 0

    def __call__(self, layout) -> float:
        self.count += 1
        return self.objective.evaluate(layout)

    @property
    def exhausted(self) -> bool:
        return self.count >= self.budget


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeterministicPlacementResult:
    """Outcome of one run. ``layout[q]`` is qubit q's ``(row, col)`` site index."""

    layout: list[tuple[int, int]]
    objective: float              # the cost's value (F_time, microseconds) for the layout
    evaluations: int
    budget: int
    route: str                    # "star", "path" or "search"
    stop_reason: str
    best_seed: str | None = None            # the cheapest seed before the search
    best_seed_objective: float | None = None
    origin_seed: str | None = None          # the seed whose descent gave the returned layout
    # schedule_screen: (name, layout) for main to schedule; the result first,
    # then the current-code pick, then the best seeds by the cost.
    candidates: tuple = ()
    seeds_scored: int = 0
    descents: int = 0
    symmetry_group_order: int | None = None


# ---------------------------------------------------------------------------
# 0. Presolve: closed forms
# ---------------------------------------------------------------------------

def _fill_idle(layout: list, R: int, C: int) -> Layout:
    """Give qubits without a site the free sites in row-major order (the cost ignores them)."""
    used = {site for site in layout if site is not None}
    free = (site for site in ((r, c) for r in range(R) for c in range(C)) if site not in used)
    return [site if site is not None else next(free) for site in layout]


def _star_layout(layer_pairs, n, R, C, distance) -> Layout | None:
    """One gate per layer, all through one hub: a proved global optimum.

    Every layer is then one load out to one stop and back, so the cost is a
    constant plus ``sum_g distance(v_g)``, which separates over leaves. For
    each hub site the leaves take the nearest sites, the most frequent leaves
    the closest ones (rearrangement inequality); the best hub site wins.
    Leaves are weighted by gate count, so a repeated leaf is placed once.
    ``distance(d_row, d_col)`` is the cost's per-gate price of a site offset:
    the round-trip move time for F_time.
    """
    gates = [gate for gates in layer_pairs for gate in gates]
    common = set(gates[0])
    for gate in gates[1:]:
        common &= set(gate)
    if not common:
        return None
    hub = min(common)
    weight = collections.Counter(b if a == hub else a for a, b in gates)
    leaves = sorted(weight, key=lambda q: (-weight[q], q))
    sites = [(r, c) for r in range(R) for c in range(C)]
    best = None
    for h in sites:
        near = sorted((distance(s[0] - h[0], s[1] - h[1]), s) for s in sites if s != h)
        value = sum(weight[q] * d for q, (d, _) in zip(leaves, near))
        if best is None or value < best[0] - _TOL:
            best = (value, h, [s for _, s in near[:len(leaves)]])
    _, h, leaf_sites = best
    layout = [None] * n
    layout[hub] = h
    for q, site in zip(leaves, leaf_sites):
        layout[q] = site
    return _fill_idle(layout, R, C)


def _path_layout(layer_pairs, n, R, C) -> Layout | None:
    """One gate per layer on a path graph: a serpentine puts every gate at the minimum displacement.

    Consecutive serpentine sites are adjacent, so every gate spans one site
    (2 grid steps), the shortest possible move, and each single-gate layer
    costs its minimum. Requires a true path (connected, acyclic), which the
    serpentine lays out end to end.
    """
    adjacency = collections.defaultdict(set)
    for gates in layer_pairs:
        for a, b in gates:
            adjacency[a].add(b)
            adjacency[b].add(a)
    nodes = list(adjacency)
    edges = sum(len(v) for v in adjacency.values()) // 2
    if any(len(v) > 2 for v in adjacency.values()) or edges != len(nodes) - 1:
        return None
    ends = sorted(q for q in nodes if len(adjacency[q]) == 1)
    order, seen = [ends[0]], {ends[0]}
    while True:
        nxt = [q for q in sorted(adjacency[order[-1]]) if q not in seen]
        if not nxt:
            break
        order.append(nxt[0])
        seen.add(nxt[0])
    if len(order) != len(nodes):
        return None                                   # disconnected
    layout = [None] * n
    for slot, q in enumerate(order):
        row, col = divmod(slot, C)
        layout[q] = (row, C - 1 - col if row % 2 else col)
    return _fill_idle(layout, R, C)


# ---------------------------------------------------------------------------
# 1a. General structured seeds: graph orders x geometric paths
# ---------------------------------------------------------------------------

def _interaction_matrix(layer_pairs, n):
    W = np.zeros((n, n))
    for gates in layer_pairs:
        for a, b in gates:
            W[a, b] += 1.0
            W[b, a] += 1.0
    return W


def _order_spectral(W):
    """Fiedler ordering (eigenvector of the second-smallest Laplacian eigenvalue)."""
    laplacian = np.diag(W.sum(1)) - W
    values, vectors = np.linalg.eigh(laplacian)
    for k in np.argsort(values, kind="stable")[1:]:
        f = vectors[:, k]
        if f.std() > 1e-12:
            return [int(q) for q in np.argsort(f, kind="stable")]
    return list(range(W.shape[0]))


def _order_reverse_cuthill_mckee(W):
    n = W.shape[0]
    degree = (W > 0).sum(1)
    seen, out = [False] * n, []
    while len(out) < n:
        start = min((i for i in range(n) if not seen[i]), key=lambda i: (degree[i], i))
        queue = collections.deque([start])
        seen[start] = True
        while queue:
            v = queue.popleft()
            out.append(v)
            neighbours = [int(u) for u in np.argsort(-W[v], kind="stable") if W[v, u] > 0 and not seen[u]]
            neighbours.sort(key=lambda u: degree[u])
            for u in neighbours:
                seen[u] = True
                queue.append(u)
    return out[::-1]


def _orders(W):
    return {
        "identity": list(range(W.shape[0])),
        "spectral": _order_spectral(W),
        "rcm": _order_reverse_cuthill_mckee(W),
        "degree": [int(q) for q in np.argsort(-W.sum(1), kind="stable")],
    }


def _serpentine_paths(n, R, C):
    """Serpentine and raster paths of every width, along either axis."""
    out = {}
    for width in range(1, max(R, C) + 1):
        for snake in (False, True):
            for transpose in (False, True):
                height, span = (C, R) if transpose else (R, C)
                if width > span or -(-n // width) > height:
                    continue
                path = []
                for slot in range(n):
                    a, b = divmod(slot, width)
                    if snake and a % 2:
                        b = width - 1 - b
                    path.append((b, a) if transpose else (a, b))
                out[f"w={width} snake={int(snake)} T={int(transpose)}"] = path
    return out


def _radial_paths(n, R, C):
    """Sites by L1 / L2 distance from each centre in one quadrant (D4 covers the rest)."""
    out = {}
    sites = [(r, c) for r in range(R) for c in range(C)]
    for metric in ("l1", "l2"):
        for cr in range((R + 1) // 2):
            for cc in range((C + 1) // 2):
                if metric == "l1":
                    key = lambda s: (abs(s[0] - cr) + abs(s[1] - cc), s)
                else:
                    key = lambda s: ((s[0] - cr) ** 2 + (s[1] - cc) ** 2, s)
                out[f"radial{metric} c=({cr},{cc})"] = sorted(sites, key=key)[:n]
    return out


def _general_seeds(layer_pairs, n, R, C):
    orders = _orders(_interaction_matrix(layer_pairs, n))
    paths = {**_serpentine_paths(n, R, C), **_radial_paths(n, R, C)}
    out = []
    for order_name, order in orders.items():
        for path_name, path in paths.items():
            if len(path) != n:
                continue
            layout = [None] * n
            for slot, q in enumerate(order):
                layout[q] = path[slot]
            out.append((f"order={order_name} | {path_name}", layout))
    return out


def _supplied_seeds(paths, n, R, C):
    """``seed_layouts`` files as seeds, checked like ``initial_layout``.

    A file lists one ``[row, col]`` SLM site per qubit; the first ``n``
    entries are used, as ``initial_layout_fill`` does.
    """
    out = []
    for path in paths:
        name = Path(path).name
        with open(path, encoding="utf-8") as f:
            sites = json.load(f)
        if not isinstance(sites, list) or len(sites) < n:
            raise ValueError(f"Seed layout {name} must list a [row, col] site for each of the {n} qubits.")
        layout = []
        for q, site in enumerate(sites[:n]):
            if (not isinstance(site, list) or len(site) != 2
                    or not all(isinstance(v, int) and not isinstance(v, bool) for v in site)
                    or not (0 <= site[0] < R and 0 <= site[1] < C)):
                raise ValueError(
                    f"Seed layout {name}: qubit {q} has {site!r}, not a [row, col] site of the {R}x{C} array."
                )
            layout.append(tuple(site))
        if len(set(layout)) != n:
            raise ValueError(f"Seed layout {name} puts two qubits on one site.")
        out.append((f"given:{name}", layout))
    return out


# ---------------------------------------------------------------------------
# 1b. Symmetry seeds: discovered translation group realised on the array
# ---------------------------------------------------------------------------

class _BlockGraph:
    """Two-qubit interaction multigraph with edges labelled by block.

    A block is a maximal run of consecutive two-qubit gates in program order
    (for a syndrome circuit, one check round); any other operation closes it.
    """

    def __init__(self, operations: Iterable[tuple[str, tuple[int, ...]]]):
        self.edges = []
        block, open_block = 0, False
        for name, qubits in operations:
            if len(qubits) == 2 and name != "barrier":
                self.edges.append((qubits[0], qubits[1], block))
                open_block = True
            elif open_block:
                block += 1
                open_block = False
        self.nblocks = block + int(open_block)
        self.support = sorted({q for a, b, _ in self.edges for q in (a, b)})
        self.incident = collections.defaultdict(list)
        for e, (a, b, _) in enumerate(self.edges):
            self.incident[a].append(e)
            self.incident[b].append(e)

    def other(self, e, v):
        a, b, _ = self.edges[e]
        return b if a == v else a


# Refinement work allowed to the symmetry search, in (rounds x edges); one
# colour-refinement round costs O(|E|). Deterministic and size-fair.
_WORK_OPS = 50_000_000
_SEARCH_NODES = 20_000


def _block_degree_signature(g: _BlockGraph):
    signature = {v: [0] * g.nblocks for v in g.support}
    for a, b, block in g.edges:
        signature[a][block] += 1
        signature[b][block] += 1
    return {v: tuple(s) for v, s in signature.items()}


def _refine(g: _BlockGraph, colours, work, rounds=60):
    """1-dimensional Weisfeiler-Leman refinement on the block-labelled multigraph."""
    colours = dict(colours)
    classes = len(set(colours.values()))
    for _ in range(rounds):
        work[0] += 1
        signature = {
            v: (colours[v], tuple(sorted((g.edges[e][2], colours[g.other(e, v)]) for e in g.incident[v])))
            for v in g.support
        }
        codes = {s: i for i, s in enumerate(sorted(set(signature.values()), key=repr))}
        colours = {v: codes[s] for v, s in signature.items()}
        if len(codes) == classes:
            break
        classes = len(codes)
    return colours


def _colour_classes(colours, support):
    out = collections.defaultdict(list)
    for v in support:
        out[colours[v]].append(v)
    return out


def _is_automorphism(g: _BlockGraph, sigma, edge_set):
    if len(set(sigma.values())) != len(g.support):
        return False
    return all((min(sigma[a], sigma[b]), max(sigma[a], sigma[b]), k) in edge_set for a, b, k in g.edges)


def _find_translations(g: _BlockGraph):
    """Fixed-point-free automorphisms, by individualisation-refinement.

    For each candidate image y of a root x with the same colour, tag x and y,
    refine both colourings, and branch on the smallest non-singleton class
    until the map is determined. Maps with a fixed point are pruned.
    """
    if not g.support:
        return []
    work, nodes = [0], [0]
    work_budget = max(64, _WORK_OPS // max(1, len(g.edges)))
    signature = _block_degree_signature(g)
    codes = {s: i for i, s in enumerate(sorted(set(signature.values()), key=repr))}
    base = _refine(g, {v: codes[s] for v, s in signature.items()}, work)
    edge_set = {(min(a, b), max(a, b), k) for a, b, k in g.edges}

    def tag(pairs, side):
        colours = dict(base)
        top = max(base.values()) + 1
        for i, pair in enumerate(pairs):
            colours[pair[side]] = top + i
        return colours

    def search(pairs):
        nodes[0] += 1
        if nodes[0] > _SEARCH_NODES or work[0] > work_budget:
            return None
        ca = _colour_classes(_refine(g, tag(pairs, 0), work), g.support)
        cb = _colour_classes(_refine(g, tag(pairs, 1), work), g.support)
        if set(ca) != set(cb) or any(len(ca[k]) != len(cb[k]) for k in ca):
            return None
        branch = None
        for k, vs in ca.items():
            if len(vs) > 1 and (branch is None or len(vs) < len(ca[branch])):
                branch = k
        if branch is None:
            sigma = {ca[k][0]: cb[k][0] for k in ca}
            if any(sigma[v] == v for v in g.support):
                return None
            return sigma if _is_automorphism(g, sigma, edge_set) else None
        x = ca[branch][0]
        for y in cb[branch]:
            if x != y:
                found = search(pairs + [(x, y)])
                if found is not None:
                    return found
        return None

    root = g.support[0]
    out = []
    for image in g.support:
        if image == root or base[image] != base[root]:
            continue
        sigma = search([(root, image)])
        if sigma is not None:
            out.append(sigma)
        if nodes[0] > _SEARCH_NODES or work[0] > work_budget:
            break
    return out


def _compose(p, q):
    return p[q]


def _generate(generators, n, cap):
    """Closure of the generators as {bytes: permutation}, or None past ``cap``."""
    identity = np.arange(n, dtype=np.int32)
    group = {identity.tobytes(): identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for h in frontier:
            for gen in generators:
                x = _compose(gen, h)
                key = x.tobytes()
                if key not in group:
                    group[key] = x
                    nxt.append(x)
                    if len(group) > cap:
                        return None
        frontier = nxt
    return group


def _free_abelian_subgroup(g: _BlockGraph, translations, top=24):
    """Largest group from <= 2 commuting discovered maps acting freely, or None."""
    n = len(g.support)
    position = {v: i for i, v in enumerate(g.support)}
    identity = np.arange(n, dtype=np.int32)
    candidates = {}
    for sigma in translations:
        p = np.array([position[sigma[v]] for v in g.support], dtype=np.int32)
        candidates[p.tobytes()] = p
    candidates.pop(identity.tobytes(), None)
    if not candidates:
        return None

    def free_closure(generators):
        group = _generate(generators, n, cap=4 * n)
        if group is None:
            return None
        for key, h in group.items():
            if key != identity.tobytes() and np.any(h == identity):
                return None
        return group

    # Keep generators whose own cyclic group is free, then rank by its size.
    viable = []
    for key in sorted(candidates):
        group = free_closure([candidates[key]])
        if group is not None:
            viable.append((candidates[key], group))
    viable.sort(key=lambda item: (-len(item[1]), item[0].tobytes()))
    best = None
    for g1, h1 in viable[:top]:
        if best is None or len(h1) > len(best):
            best = h1
        for g2, _ in viable:
            if g2.tobytes() in h1 or not np.array_equal(_compose(g1, g2), _compose(g2, g1)):
                continue
            h2 = free_closure([g1, g2])
            if h2 is not None and len(h2) > len(best):
                best = h2
    if best is None:
        return None
    group = list(best.values())
    seen, orbits = set(), []
    for i in range(n):
        if i not in seen:
            orbit = sorted({int(h[i]) for h in group})
            orbits.append(orbit)
            seen |= set(orbit)
    return {"group": group, "size": len(group), "orbits": orbits}


def _presentations(group_info):
    """Every shape (a, b) with H = <g1> x <g2>, |g1| = a, |g2| = b."""
    group = group_info["group"]
    n, size = len(group[0]), len(group)
    identity_key = np.arange(n, dtype=np.int32).tobytes()
    cyclic = {}
    for p in group:
        if p.tobytes() != identity_key:
            cyclic[p.tobytes()] = set(_generate([p], n, cap=size))
    out = {}
    for k1, c1 in cyclic.items():
        for k2, c2 in cyclic.items():
            if len(c1) * len(c2) == size and len(c1 & c2) == 1:
                out.setdefault((len(c1), len(c2)), (k1, k2))
    return out


# Cap on torus seeds (binding only with torus_all_pairs). A cap of 12,000 would
# cut off bb144's best torus seed on a 30x30 array (about the 13,000th generated).
_TORUS_ALL_PAIRS_CAP = 50_000


def _generator_pairs(group_info):
    """Every generator pair per shape: {(L, M): [(g1 key, g2 key), ...]} with H = <g1> x <g2>.

    ``_presentations`` keeps only the first pair per shape.
    """
    group = group_info["group"]
    size = group_info["size"]
    n = len(group[0])
    identity = np.arange(n, dtype=np.int32).tobytes()
    cyclic = {p.tobytes(): set(_generate([p], n, cap=size)) for p in group if p.tobytes() != identity}
    out = {}
    for k1, c1 in cyclic.items():
        for k2, c2 in cyclic.items():
            shape = (len(c1), len(c2))
            if shape[0] * shape[1] == size and len(c1 & c2) == 1:
                out.setdefault(shape, []).append((k1, k2))
    return out


def _symmetry_seeds(g: _BlockGraph, n, R, C, all_pairs=False, cap=_TORUS_ALL_PAIRS_CAP):
    """Equivariant layouts: every gate of a symmetry orbit gets one displacement.

    For each presentation H = Z_L x Z_M, group element g1^i g2^j becomes torus
    point (i, j); each of at most 4 qubit orbits takes one cell of a 2x2
    block, so orbit member h(rep) sits at site (2i + dr, 2j + dc). All 4!
    orbit-to-cell assignments and both axis orders are generated; layouts
    must fit the array. Qubits outside the circuit's support take spare sites
    in row-major order.

    One generator pair per shape by default; with ``all_pairs`` every pair
    (``_generator_pairs``), at most ``cap`` layouts. The pair changes which
    group elements share a row, and on bb144 the best pair is not the first.
    Returns ``(seeds, group order or None)``.
    """
    translations = _find_translations(g)
    info = _free_abelian_subgroup(g, translations) if translations else None
    if info is None or len(info["orbits"]) > 4:
        return [], (info["size"] if info else None)
    by_key = {h.tobytes(): h for h in info["group"]}
    support = g.support
    out, seen = [], set()
    if all_pairs:
        pairs = [(shape, pair) for shape, plist in sorted(_generator_pairs(info).items()) for pair in plist]
    else:
        pairs = sorted(_presentations(info).items(), key=lambda kv: repr(kv[0]))
    for (L, M), (k1, k2) in pairs:
        if len(out) >= cap:
            break
        g1, g2 = by_key[k1], by_key[k2]
        coord = {}
        current_i = np.arange(len(support), dtype=np.int32)
        for i in range(L):
            current = current_i
            for j in range(M):
                coord[current.tobytes()] = (i, j)
                current = _compose(g2, current)
            current_i = _compose(g1, current_i)
        if len(coord) != info["size"]:
            continue
        for cells in itertools.permutations(range(4)):
            for swap in (False, True):
                shape = (M, L) if swap else (L, M)
                if 2 * shape[0] > R or 2 * shape[1] > C:
                    continue
                placed = {}
                for k, orbit in enumerate(info["orbits"]):
                    dr, dc = divmod(cells[k], 2)
                    for h in info["group"]:
                        i, j = coord[h.tobytes()]
                        if swap:
                            i, j = j, i
                        placed[support[int(h[orbit[0]])]] = (2 * i + dr, 2 * j + dc)
                if len(placed) != len(support) or len(set(placed.values())) != len(placed):
                    continue
                if any(q >= n for q in placed):
                    continue
                layout = _fill_idle([placed.get(q) for q in range(n)], R, C)
                key = tuple(layout)
                if key not in seen:
                    seen.add(key)
                    out.append((f"torus {shape[0]}x{shape[1]}/c{''.join(map(str, cells))}s{int(swap)}", layout))
    return out[:cap], info["size"]


# ---------------------------------------------------------------------------
# 1c. Product seeds: hypergraph-product codes on their two factor coordinates
# (``product_seeds``).
# ---------------------------------------------------------------------------

# Factors with at most this many vertices have every vertex order scored
# (10! orders take a few seconds). Larger factors use the spectral, reverse
# Cuthill-McKee and identity orders instead.
_PRODUCT_ENUMERATION_MAX = 10


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            if a > b:
                a, b = b, a
            self.parent[b] = a


def _mixed_squares(g: _BlockGraph, same_block=False):
    """4-cycles u-a-w-b as edge ids (e_ua, e_ub, e_wa, e_wb), u < w, a < b.

    Mixed: u's two edges in one block and w's in another (X check - data -
    Z check - data). ``same_block`` (a one-block graph): every 4-cycle.
    """
    by_block = collections.defaultdict(lambda: collections.defaultdict(list))
    for e, (x, y, k) in enumerate(g.edges):
        by_block[x][k].append((y, e))
        by_block[y][k].append((x, e))
    out = []
    for u in g.support:
        for bu, edges_u in sorted(by_block[u].items()):
            bucket = collections.defaultdict(list)
            for a, e_ua in edges_u:
                for bw, edges_a in by_block[a].items():
                    if bw == bu and not same_block:
                        continue
                    for w, e_aw in edges_a:
                        if w > u:
                            bucket[(w, bw)].append((a, e_ua, e_aw))
            for _, entries in sorted(bucket.items()):
                entries.sort()
                for i in range(len(entries)):
                    for j in range(i + 1, len(entries)):
                        a, e_ua, e_wa = entries[i]
                        b, e_ub, e_wb = entries[j]
                        if a != b:
                            out.append((e_ua, e_ub, e_wa, e_wb))
    return out


def _product_structure(g: _BlockGraph, n):
    """Certified Cartesian product of the interaction graph, or ``{"is_product": False}``.

    Opposite edges of every mixed square fall in one parallel class (union
    find); classes 2-coloured by orthogonality split into two factors; a
    qubit's coordinate in factor f is its component in the other factor's
    edges. Certified when coordinates are injective, every class is one
    factor edge, and every class is a matching.
    """
    E = len(g.edges)
    squares = _mixed_squares(g, same_block=(g.nblocks == 1))
    record = {"is_product": False}
    if not squares:
        return record
    uf = _UnionFind(E)
    for e_ua, e_ub, e_wa, e_wb in squares:
        uf.union(e_ua, e_wb)
        uf.union(e_ub, e_wa)
    roots = [uf.find(e) for e in range(E)]
    ids = {r: i for i, r in enumerate(sorted(set(roots)))}
    cls = np.array([ids[r] for r in roots], np.int64)
    on_square = np.zeros(E, bool)
    for square in squares:
        on_square[list(square)] = True
    orth, self_orth = collections.defaultdict(set), False
    for e_ua, e_ub, e_wa, e_wb in squares:
        for x, y in ((e_ua, e_ub), (e_wa, e_wb)):
            cx, cy = int(cls[x]), int(cls[y])
            self_orth |= cx == cy
            orth[cx].add(cy)
            orth[cy].add(cx)
    factor = np.full(len(ids), -1, np.int64)
    bipartite = not self_orth
    for start in sorted(orth):
        if factor[start] >= 0:
            continue
        factor[start], stack = 0, [start]
        while stack and bipartite:
            x = stack.pop()
            for y in sorted(orth[x]):
                if factor[y] < 0:
                    factor[y] = 1 - factor[x]
                    stack.append(y)
                elif factor[y] == factor[x]:
                    bipartite = False
                    break
    if not bipartite or not on_square.all():
        return record
    matching = True
    for c in range(len(ids)):
        seen = set()
        for e in np.flatnonzero(cls == c).tolist():
            a, b, _ = g.edges[e]
            if a in seen or b in seen:
                matching = False
            seen.update((a, b))
    edge_factor = factor[cls]
    coords = np.full((n, 2), -1, np.int64)
    for f in (0, 1):
        uf2 = _UnionFind(n)
        for e in range(E):
            if edge_factor[e] != f:
                a, b, _ = g.edges[e]
                uf2.union(a, b)
        component = {r: i for i, r in enumerate(sorted({uf2.find(v) for v in g.support}))}
        for v in g.support:
            coords[v, f] = component[uf2.find(v)]
    injective = len({tuple(coords[v]) for v in g.support}) == len(g.support)
    single = True
    for c in range(len(ids)):
        f = int(factor[c])
        pairs = set()
        for e in np.flatnonzero(cls == c).tolist():
            a, b, _ = g.edges[e]
            if coords[a, 1 - f] != coords[b, 1 - f]:
                single = False
                break
            pairs.add(tuple(sorted((int(coords[a, f]), int(coords[b, f])))))
        if not single or len(pairs) != 1:
            single = False
            break
    return {"is_product": bool(injective and single and matching), "coords": coords}


def _factor_edges(g: _BlockGraph, coords, f):
    """Factor-f edges (coordinate pairs) of the gates whose other coordinate is constant."""
    out = set()
    for a, b, _ in g.edges:
        if coords[a, 1 - f] == coords[b, 1 - f] and coords[a, f] != coords[b, f]:
            out.add(tuple(sorted((int(coords[a, f]), int(coords[b, f])))))
    return sorted(out)


def _linear_order(nv, edges):
    """Vertex order of a path factor (from its smaller end), or None if the factor is not a path."""
    adjacency = collections.defaultdict(set)
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    if any(len(adjacency[v]) > 2 for v in range(nv)):
        return None
    ends = [v for v in range(nv) if len(adjacency[v]) == 1]
    if len(ends) != 2:
        return None
    order, previous = [min(ends)], None
    while len(order) < nv:
        following = [w for w in adjacency[order[-1]] if w != previous]
        if not following:
            return None
        previous = order[-1]
        order.append(min(following))
    return order


def _product_factor_orders(nv, edges, move_seconds, top_k=4):
    """The ``top_k`` best vertex orders of one factor: [(score, order)], best first.

    A path factor has one order, along the path. Otherwise every order is
    scored when ``nv <= _PRODUCT_ENUMERATION_MAX``, else the spectral,
    reverse Cuthill-McKee and identity orders. Score: each distinct signed span
    of the factor's edges (side 0 -> side 1 of its 2-colouring) costs one move
    of that many trap steps. Returns [] when the factor is not bipartite.
    """
    path = _linear_order(nv, edges)
    adjacency = [[] for _ in range(nv)]
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    side = [-1] * nv
    for start in range(nv):
        if side[start] >= 0:
            continue
        side[start], stack = 0, [start]
        while stack:
            u = stack.pop()
            for w in adjacency[u]:
                if side[w] < 0:
                    side[w] = 1 - side[u]
                    stack.append(w)
                elif side[w] == side[u]:
                    return []
    directed = np.array([(a, b) if side[a] == 0 else (b, a) for a, b in edges], np.int64).reshape(-1, 2)
    span_seconds = np.array([move_seconds(abs(k)) for k in range(-nv, nv + 1)])

    def score(inverse):
        spans = inverse[:, directed[:, 1]] - inverse[:, directed[:, 0]]
        present = np.zeros((len(inverse), 2 * nv + 1), bool)
        np.put_along_axis(present, spans + nv, True, axis=1)
        return present @ span_seconds + 1e-9 * present.sum(1)

    def ranked(orders):
        block = np.array(orders, np.int64)
        values = score(np.argsort(block, axis=1))
        return sorted((float(v), list(map(int, o))) for v, o in zip(values, block))[:top_k]

    if path is not None:
        return ranked([path])
    if nv <= _PRODUCT_ENUMERATION_MAX:
        best, orders = [], itertools.permutations(range(nv))
        while True:
            chunk = list(itertools.islice(orders, 400_000))
            if not chunk:
                return best
            best = sorted(best + ranked(chunk))[:top_k]
    W = np.zeros((nv, nv))
    for a, b in edges:
        W[a, b] += 1.0
        W[b, a] += 1.0
    candidates = {tuple(range(nv)), tuple(_order_spectral(W)), tuple(_order_reverse_cuthill_mckee(W))}
    return ranked(sorted(candidates))


def _grid_images(sites, R, C):
    """The up to 8 reflections / transposes of ``sites`` that fit the R x C array, deduplicated."""
    t = np.asarray(sites, np.int64)
    rmax, cmax = int(t[:, 0].max()), int(t[:, 1].max())
    out, seen = [], set()
    for transpose, flip_r, flip_c in itertools.product((False, True), repeat=3):
        r, c = (t[:, 1], t[:, 0]) if transpose else (t[:, 0], t[:, 1])
        hr, hc = (cmax, rmax) if transpose else (rmax, cmax)
        r = hr - r if flip_r else r
        c = hc - c if flip_c else c
        if r.max() >= R or c.max() >= C:
            continue
        key = tuple(zip(r.tolist(), c.tolist()))
        if key not in seen:
            seen.add(key)
            out.append((f"img{'t' if transpose else ''}{'r' if flip_r else ''}{'c' if flip_c else ''}", key))
    return out


def _product_seeds(g: _BlockGraph, n, R, C, move_seconds, top_k=4):
    """Layouts of a certified Cartesian-product circuit: qubit (u, v) at site (order0(u), order1(v)).

    Every combination of each factor's ``top_k`` orders, in each of the up to
    8 array images that fit; qubits outside the circuit take spare sites in
    row-major order. [] when the circuit is not a product or nothing fits.
    """
    product = _product_structure(g, n)
    if not product["is_product"]:
        return []
    coords = product["coords"]
    support = [q for q in range(n) if coords[q, 0] >= 0]
    sizes = [int(coords[support, f].max()) + 1 for f in (0, 1)]
    scored = {}                    # identical factors (e.g. both Hamming) are ordered once
    orders = []
    for f in (0, 1):
        key = (sizes[f], tuple(_factor_edges(g, coords, f)))
        if key not in scored:
            scored[key] = _product_factor_orders(key[0], list(key[1]), move_seconds, top_k)
        orders.append(scored[key])
    if not orders[0] or not orders[1]:
        return []
    out, seen = [], set()
    for (s0, o0), (s1, o1) in itertools.product(*orders):
        p0 = {v: i for i, v in enumerate(o0)}
        p1 = {v: i for i, v in enumerate(o1)}
        base = [(p0[int(coords[q, 0])], p1[int(coords[q, 1])]) for q in support]
        for tag, image in _grid_images(base, R, C):
            placed = dict(zip(support, image))
            layout = _fill_idle([placed.get(q) for q in range(n)], R, C)
            key = tuple(layout)
            if key not in seen:
                seen.add(key)
                out.append((f"product {sizes[0]}x{sizes[1]} {tag}", layout))
    return out


# ---------------------------------------------------------------------------
# 2. Search: candidate-list descent, layer-guided descent, VND, multistart
# ---------------------------------------------------------------------------

def _candidate_lists(n, adjacency, layout, k_geom, k_disc):
    """Interaction neighbours + k nearest qubits + k_disc well-spread qubits.

    The spread picks walk each qubit's distance-ordered list at golden-ratio
    strides (offset by the qubit index), a deterministic low-discrepancy
    sample that lets every qubit see a few distant partners.
    """
    points = np.asarray(layout, dtype=np.int64)
    d2 = ((points[:, None, 0] - points[None, :, 0]) ** 2
          + (points[:, None, 1] - points[None, :, 1]) ** 2)
    order = np.argsort(d2, axis=1, kind="stable")
    out = {}
    for i in range(n):
        near = order[i].tolist()
        pick = set(adjacency[i]) | set(near[1:k_geom + 1])
        for m in range(1, k_disc + 1):
            idx = int((((i + m) * _PHI_INV) % 1.0) * (n - 1)) + 1
            if idx < len(near):
                pick.add(near[idx])
        out[i] = sorted(pick - {i})
    return out


def _free_sites(layout, R, C):
    occupied = set(layout)
    return [(r, c) for r in range(R) for c in range(C) if (r, c) not in occupied]


def _descent_swap(evaluate: _Evaluator, layout, R, C, adjacency, k_geom, k_disc, free_per_qubit):
    """Best-improvement swap/relocate descent with candidate lists and a don't-look queue.

    Qubits wait in a FIFO queue. For the popped qubit, try a swap with each
    candidate and a move to each of the ``free_per_qubit`` nearest free
    sites; take the best improving one. After an accepted move, re-queue the
    moved qubits and their interaction neighbours (Bentley's don't-look bits,
    widened because the cost is layer-global, not a sum of edge costs). Ends when
    the queue empties: a local optimum of this restricted neighbourhood.
    """
    n = len(layout)
    current = list(layout)
    candidates = _candidate_lists(n, adjacency, current, k_geom, k_disc)
    best = evaluate(current)
    free = _free_sites(current, R, C)
    queue = collections.deque(range(n))
    queued = [True] * n
    while queue and not evaluate.exhausted:
        i = queue.popleft()
        queued[i] = False
        local_best, local_layout, moved = best, None, ()
        for j in candidates[i]:
            if evaluate.exhausted:
                break
            trial = list(current)
            trial[i], trial[j] = trial[j], trial[i]
            value = evaluate(trial)
            if value < local_best - 1e-15:
                local_best, local_layout, moved = value, trial, (i, j)
        if free:
            ri, ci = current[i]
            nearest = sorted(free, key=lambda s: ((s[0] - ri) ** 2 + (s[1] - ci) ** 2, s))[:free_per_qubit]
            for site in nearest:
                if evaluate.exhausted:
                    break
                trial = list(current)
                trial[i] = site
                value = evaluate(trial)
                if value < local_best - 1e-15:
                    local_best, local_layout, moved = value, trial, (i,)
        if local_layout is None:
            continue
        current, best = local_layout, local_best
        free = _free_sites(current, R, C)
        wake = set(moved)
        for q in moved:
            wake |= adjacency[q]
        for q in wake:
            if not queued[q]:
                queued[q] = True
                queue.append(q)
    return best, current


def _canonical(v):
    """One representative of ``{v, -v}``: a gate's displacement up to which operand moves."""
    x, y = v
    return (-x, -y) if x < 0 or (x == 0 and y < 0) else (x, y)


def _descent_layer(evaluate: _Evaluator, layout, R, C, layer_pairs, max_rounds=200):
    """Best-improvement descent over layer-guided moves.

    The cost charges a pulse and a tour stop per distinct displacement. So
    for every gate whose displacement differs from its layer's majority m,
    try moving one endpoint to (other endpoint +- m), swapping with the
    occupant if that site is taken.
    """
    current = list(layout)
    best = evaluate(current)
    for _ in range(max_rounds):
        if evaluate.exhausted:
            break
        occupied = {site: q for q, site in enumerate(current)}
        moves = []
        for gates in layer_pairs:
            if len(gates) < 2:
                continue
            displacement = [_canonical((current[b][0] - current[a][0], current[b][1] - current[a][1]))
                            for a, b in gates]
            counts = collections.Counter(displacement)
            if len(counts) < 2:
                continue
            majority = counts.most_common(1)[0][0]
            for (a, b), d in zip(gates, displacement):
                if d == majority:
                    continue
                for sign in (1, -1):
                    for mover, anchor in ((b, a), (a, b)):
                        target = (current[anchor][0] + sign * majority[0],
                                  current[anchor][1] + sign * majority[1])
                        if 0 <= target[0] < R and 0 <= target[1] < C and target != current[mover]:
                            moves.append((mover, target, occupied.get(target)))
        round_best, round_layout, seen = best, None, set()
        for mover, target, occupant in moves:
            if evaluate.exhausted:
                break
            if (mover, target) in seen:
                continue
            seen.add((mover, target))
            trial = list(current)
            if occupant is None:
                trial[mover] = target
            else:
                trial[mover], trial[occupant] = trial[occupant], trial[mover]
            value = evaluate(trial)
            if value < round_best - 1e-15:
                round_best, round_layout = value, trial
        if round_layout is None:
            break
        current, best = round_layout, round_best
    return best, current


def _vnd(evaluate: _Evaluator, layout, R, C, adjacency, layer_pairs, options):
    """Variable neighbourhood descent: swap(k) / layer moves for k in the ladder.

    Restart at the first neighbourhood after any improvement, otherwise
    advance; stop when the last one fails or the budget runs out.
    """
    phases = []
    for k in options["k_ladder"]:
        phases.append(("swap", k))
        if options["layer_moves"]:
            phases.append(("layer", k))
    current, best, i = list(layout), None, 0
    while i < len(phases) and not evaluate.exhausted:
        kind, k = phases[i]
        if kind == "swap":
            value, result = _descent_swap(evaluate, current, R, C, adjacency, k,
                                          options["k_disc"], options["free_per_qubit"])
        else:
            value, result = _descent_layer(evaluate, current, R, C, layer_pairs)
        if best is None:
            best, current, i = value, result, i + 1
        elif value < best - _TOL:
            best, current, i = value, result, 0
        else:
            i += 1
    return best, current


def _multistart(evaluate: _Evaluator, ranked, R, C, adjacency, layer_pairs, options):
    """Best-first descent of the ranked seeds under a diversity veto and stall rule.

    Returns the best value and layout, the number of descents, the stop
    reason, and the index into ``ranked`` of the seed the best layout came from.
    """
    n = len(ranked[0][2])
    min_sep = options["min_sep"] or max(2, n // 6)
    min_spend = int(options["min_spend_fraction"] * evaluate.budget)
    heap = [(value, index, tuple(layout)) for index, (value, _, layout) in enumerate(ranked)]
    heapq.heapify(heap)
    descended, best, best_layout, origin = [], None, None, None
    descents = since_improve = 0
    reason = None
    while heap and not evaluate.exhausted:
        _, index, layout = heapq.heappop(heap)
        if any(sum(x != y for x, y in zip(layout, d)) < min_sep for d in descended):
            continue
        descended.append(layout)
        descents += 1
        value, end = _vnd(evaluate, list(layout), R, C, adjacency, layer_pairs, options)
        descended.append(tuple(end))
        improved = best is None or value < best - _TOL
        if improved:
            best, best_layout, origin = value, end, index
        if evaluate.exhausted:
            reason = "budget cap"
            break
        if improved:
            since_improve = 0
        else:
            since_improve += 1
            if options["stall"] is not None and evaluate.count >= min_spend and since_improve >= options["stall"]:
                reason = f"stall ({options['stall']} descent(s) without improvement)"
                break
    if reason is None:
        reason = "budget cap" if evaluate.exhausted else "exhaustion (every seed descended or vetoed)"
    return best, best_layout, descents, reason, origin


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _program_operations(two_qubit_dag, reference_nodes):
    """(name, qubit ids) in program order (topological without reference_nodes), for symmetry discovery."""
    nodes = reference_nodes if reference_nodes is not None else two_qubit_dag.topological_op_nodes()
    for node in nodes:
        yield node.op.name, tuple(extract_index_from_bit(q) for q in node.qargs)


def run_deterministic_placement(
    two_qubit_dag,
    num_qubits: int,
    dimensions: Sequence[int],
    *,
    reference_nodes: Sequence[Any] | None = None,
    options: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    cost=None,
) -> DeterministicPlacementResult:
    """Minimise F_time deterministically; see the module docstring.

    ``reference_nodes`` (program-order op nodes, as ``main`` loads them) give
    symmetry discovery its gate blocks and F_time its reorder_cz sections;
    without them blocks come from the two-qubit DAG (only barriers split
    them). ``config`` supplies the physics F_time needs. ``cost``
    substitutes any object with ``evaluate``, ``layer_pairs`` and
    ``star_distance`` (Legacy/cost_fidelity.py passes the legacy F from
    Legacy/legacy_f.py); ``config`` is then not needed.
    """
    options = validate_options(options)
    R, C = int(dimensions[0]), int(dimensions[1])
    n = int(num_qubits)
    if n > R * C:
        raise ValueError(f"{n} qubits do not fit {R}x{C} = {R * C} sites.")
    if cost is None:
        if config is None:
            raise ValueError("run_deterministic_placement needs the config's physics to evaluate F_time.")
        cost = make_cost(two_qubit_dag, n, config, reference_nodes)
    objective = cost
    layer_pairs = objective.layer_pairs
    evaluate = _Evaluator(objective, options["budget"])

    def finish(layout, route, reason, **extra):
        return DeterministicPlacementResult(
            layout=[tuple(site) for site in layout], objective=objective.evaluate(layout),
            evaluations=evaluate.count, budget=evaluate.budget,
            route=route, stop_reason=reason, **extra,
        )

    if not layer_pairs:
        return finish(_fill_idle([None] * n, R, C), "search", "no two-qubit gates")

    # 0. Presolve.
    if options["closed_forms"] and all(len(gates) == 1 for gates in layer_pairs):
        star = _star_layout(layer_pairs, n, R, C, objective.star_distance)
        if star is not None:
            evaluate(star)
            return finish(star, "star", "closed form (proved global optimum)")
        path = _path_layout(layer_pairs, n, R, C)
        if path is not None:
            evaluate(path)
            return finish(path, "path", "construction: every gate at the minimum displacement")

    # 1. Seeds, each distinct layout scored once and ranked by cost.
    groups = [[("row-major", [divmod(i, C) for i in range(n)])],
              _general_seeds(layer_pairs, n, R, C)]
    group_order = None
    current_torus = []
    if options["symmetry_seeds"]:
        graph = _BlockGraph(_program_operations(two_qubit_dag, reference_nodes))
        torus, group_order = _symmetry_seeds(graph, n, R, C, all_pairs=options["torus_all_pairs"])
        groups.append(torus)
        # The torus seeds the current code uses (one pair per shape), for the screening baseline.
        current_torus = _symmetry_seeds(graph, n, R, C)[0] if options["torus_all_pairs"] else torus
    if options["product_seeds"]:
        groups.append(_product_seeds(_BlockGraph(_program_operations(two_qubit_dag, reference_nodes)),
                                     n, R, C, _trap_move_seconds(config)))
    # Last: the ranking is stable, so a constructed seed wins any tie.
    groups.append(_supplied_seeds(options["seed_layouts"], n, R, C))
    ranked, seen = [], set()
    for group in groups:
        for name, layout in group:
            layout = [tuple(site) for site in layout]
            key = tuple(layout)
            if key in seen or evaluate.exhausted:
                continue
            seen.add(key)
            ranked.append((evaluate(layout), name, layout))
    ranked.sort(key=lambda item: item[0])          # stable: ties keep generation order
    seed_value, seed_name, seed_layout = ranked[0]

    # 2. Multistart search.
    adjacency = collections.defaultdict(set)
    for gates in layer_pairs:
        for a, b in gates:
            adjacency[a].add(b)
            adjacency[b].add(a)
    best, best_layout, descents, reason, origin = _multistart(
        evaluate, ranked, R, C, adjacency, layer_pairs, options,
    )
    origin_seed = None if origin is None else ranked[origin][1]
    if best is None or seed_value <= best:
        best_layout, origin_seed = seed_layout, seed_name
    candidates = _screen_candidates(best_layout, origin_seed, ranked, groups, current_torus, options)
    return finish(best_layout, "search", reason, best_seed=seed_name,
                  best_seed_objective=seed_value, origin_seed=origin_seed, candidates=candidates,
                  seeds_scored=len(ranked), descents=descents,
                  symmetry_group_order=group_order)


def _trap_move_seconds(config):
    """``k -> seconds`` of a straight move of k trap steps (2k grid steps); ``float`` (k itself) without config."""
    if config is None:
        return float
    from .dynamics import _motion_parameters, _step_seconds
    motion = {"_motion_parameters": config.get("_motion_parameters") or _motion_parameters(config),
              "movement_shape": config.get("movement_shape", "trapezoidal")}
    return lambda k: _step_seconds((2 * k, 0), motion)


def _screen_candidates(best_layout, origin_seed, ranked, groups, current_torus, options):
    """Layouts for main to schedule under ``schedule_screen`` (empty when it is off).

    First the search's result, then the best seed of the pool the current
    code builds (row-major, general and one-pair torus seeds, plus any
    ``seed_layouts``), so screening can only match or beat the current code;
    then the best remaining seeds by the cost. Distinct layouts, at most
    ``schedule_screen``. The cost cannot order near-equivalent layouts (on
    bb144 the fastest torus layout is about the 93rd by F_time), which is why
    main judges them by schedule time.
    """
    if not options["schedule_screen"]:
        return ()
    current = {tuple(map(tuple, layout)) for group in (groups[0], groups[1], groups[-1]) for _, layout in group}
    current |= {tuple(map(tuple, layout)) for _, layout in current_torus}
    head = [(f"search result ({origin_seed})", best_layout)]
    head += [(f"current-code pick ({name})", layout) for _, name, layout in ranked
             if tuple(map(tuple, layout)) in current][:1]
    out, seen = [], set()
    for name, layout in head + [(name, layout) for _, name, layout in ranked]:
        key = tuple(map(tuple, layout))
        if key not in seen:
            seen.add(key)
            out.append((name, [tuple(site) for site in layout]))
            if len(out) >= options["schedule_screen"]:
                break
    return tuple(out)


def deterministic_fill(
    grid,
    n: int,
    two_qubit_dag,
    config: Mapping[str, Any],
    *,
    reference_nodes: Sequence[Any] | None = None,
    options: Mapping[str, Any] | None = None,
    return_result: bool = False,
) -> list[Qubit] | tuple[list[Qubit], DeterministicPlacementResult]:
    """Place qubits with the deterministic minimiser (``fill_strategy: "deterministic"``)."""
    result = run_deterministic_placement(
        two_qubit_dag, n, config["dimensions"],
        reference_nodes=reference_nodes, options=options, config=config,
    )
    for row_nodes in grid:
        for node in row_nodes:
            node.qubit = None
    qubits = [place_qubit(grid, 2 * r, 2 * c, q) for q, (r, c) in enumerate(result.layout)]
    return (qubits, result) if return_result else qubits
