"""Deterministic initial placement: ``fill_strategy: "deterministic"``.

Ported from the collaborator's (Munson's) deterministic minimiser of the
FastSA objective. It minimises the same cost FastSA anneals, F_time
(placement_cost.py), on the same two-qubit DAG, but with no random numbers:

  0. presolve   closed forms for star and path circuits
  1. seeds      interaction-graph orders laid along geometric paths, layouts
                that realise the circuit's translation symmetry, row-major,
                and any layouts supplied with the ``seed_layouts`` option
  2. search     best-first multistart over the seeds, each descended by a
                variable neighbourhood descent (swap/relocate moves with
                candidate lists, and layer-guided moves)

Every search evaluation is counted against ``budget``. Ties are broken by
index, so a run is reproducible. Only the placement path is ported: the
original's analysis pool, antiparallel screen and disabled neighbourhoods
(ejection chains, subgroup tilings) are left out.

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
    # Layout JSON files ([row, col] site per qubit, as initial_layout) added
    # to the seed pool. They start the search from a known layout, so the
    # result reports it in origin_seed. Empty: every seed is constructed.
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
    ladder = merged["k_ladder"]
    if not isinstance(ladder, (list, tuple)) or not ladder or not all(
        isinstance(k, int) and not isinstance(k, bool) and k > 0 for k in ladder
    ):
        raise ValueError("Deterministic_fill.k_ladder must be a non-empty list of positive integers.")
    merged["k_ladder"] = list(ladder)
    for key in ("layer_moves", "symmetry_seeds", "closed_forms"):
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


def _symmetry_seeds(g: _BlockGraph, n, R, C):
    """Equivariant layouts: every gate of a symmetry orbit gets one displacement.

    For each presentation H = Z_L x Z_M, group element g1^i g2^j becomes torus
    point (i, j); each of at most 4 qubit orbits takes one cell of a 2x2
    block, so orbit member h(rep) sits at site (2i + dr, 2j + dc). All 4!
    orbit-to-cell assignments and both axis orders are generated; layouts
    must fit the array. Qubits outside the circuit's support take spare sites
    in row-major order.
    """
    translations = _find_translations(g)
    info = _free_abelian_subgroup(g, translations) if translations else None
    if info is None or len(info["orbits"]) > 4:
        return [], (info["size"] if info else None)
    by_key = {h.tobytes(): h for h in info["group"]}
    support = g.support
    out, seen = [], set()
    for (L, M), (k1, k2) in sorted(_presentations(info).items(), key=lambda kv: repr(kv[0])):
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
    return out, info["size"]


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
    """(name, qubit ids) in program order, for block-labelled symmetry discovery."""
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
    symmetry discovery its gate blocks; without them the two-qubit DAG is
    used as one block. ``config`` supplies the physics F_time needs. ``cost``
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
    if options["symmetry_seeds"]:
        graph = _BlockGraph(_program_operations(two_qubit_dag, reference_nodes))
        torus, group_order = _symmetry_seeds(graph, n, R, C)
        groups.append(torus)
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
    return finish(best_layout, "search", reason, best_seed=seed_name,
                  best_seed_objective=seed_value, origin_seed=origin_seed,
                  seeds_scored=len(ranked), descents=descents,
                  symmetry_group_order=group_order)


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
