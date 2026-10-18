# Exact displacement tours: the Noon–Bean + Jonker–Volgenant gadget

Implementation references:

- [`naive_n_dag` gadget construction and decoding](src/naive_n_dag/tsp_gadget.py)
- [`naive_n_dag` dynamics (`_order_group_displacements` and helpers)](src/naive_n_dag/dynamics.py)
- [Gadget tests](tests/test_tsp_gadget.py)

This document covers how the scheduler orders the gate visits of one AOD
load, why the original tour was not exact, the gadget that makes it exact,
the proofs, the validation applied to solver output, and the configuration.

## 1. The problem a tour solves

One AOD load picks up a rigid set of movers, visits a sequence of
interaction targets, and unloads at home. Everything is expressed relative
to the movers' homes, so each gate is a **city**: an even-even displacement
`(r, c)`, labelled with the layer it belongs to.

- A gate happens with the loaded atom at one of the **four cardinal sites** of
  its target, `(r±1, c)` or `(r, c±1)`. The atom arrives at that site, the gate
  fires, and the atom leaves **from that same site**.
- Motion between two sites follows `parity_route_moves` (the odd-row/odd-column
  highways). Its time is `_route(a, b)`, which is the same per-step time
  `schedule_duration` charges.
- **Loading from home** is one cardinal transfer step, in any of 4 directions,
  followed by a route. A **retained load** starts at its current AOD offset
  without a transfer. **Unloading** routes to any of the 4 home-adjacent sites
  and then transfers down.
- Gates in the same layer commute and may be visited in any order. Layers
  must be visited in order.

The **exact objective** is the total motion time from load to unload, minimized
jointly over the visit order and the side chosen at every city. Load/unload
steps are never merged with highway moves (`_compress_linear_moves`), and a
gate separates every pair of consecutive visits. The tour's cost is therefore
exactly the motion time the schedule pays. Transfers, pulses and gate times
are the same for every order, so they don't affect the choice.

### Why the original (relaxed) tour is not exact

`_layered_tour_graph` uses one node per city. Each edge is priced by
`_side_movement` as the cheapest route between **any** side of one city and
**any** side of the other. A tour can therefore arrive at a city on its north
side and leave from its east side, which is physically impossible.
`_optimize_visit_sides` then fixes the sides exactly, but only for the order
the relaxed TSP already chose. The relaxed optimum is a lower bound, not the
optimal routing. On 300 random instances it was suboptimal 21% of the time,
by up to 10.3%.

## 2. Per-group pipeline

`_order_group_displacements` runs this for every load (ordinary groups,
retained-load reroutes, and trial plans). The config key `tour_ordering`
selects how much of it runs:

| `tour_ordering` | Candidates priced | Relaxed solver order |
|---|---|---|
| `"tsp_gadget"` (default) | relaxed, alignment, exact | fast_tsp, then Concorde |
| `"tsp"` | relaxed, alignment | Concorde, then fast_tsp |
| `"alignment"` | alignment only; no TSP graph, solver or pricing | – |

Under every setting the side DP (`_optimize_visit_sides`) then picks the
interaction sites for the chosen order. The steps below are the
`"tsp_gadget"` pipeline; `"tsp"` skips step 3.

1. **Relaxed tour (always computed; it is the backup).** Build the side-priced
   layer ring and solve it with a two-solver chain.
   - With the gadget on, Concorde is busy with the exact tour, so **fast_tsp**
     (`tsp_time_limit`, default 0.05 s) goes first and **Concorde** backs it up.
   - Under `tour_ordering: "tsp"`, **Concorde** goes first (best tour within
     `concorde_time_limit`) and fast_tsp backs it up.

   A solver that is missing, fails, or returns an unfaithful tour hands over to
   the next one. If both fail, the alignment order stands in. Integer weights
   are sized by a known feasible tour (§6).
2. **Alignment order (always computed).** The per-layer alignment heuristic
   costs almost nothing, so it is always a candidate.
3. **Exact tour (only under `tour_ordering: "tsp_gadget"`).**
   - If the group has at most `_EXACT_ENUMERATION_LIMIT` = 24 layer-respecting
     orders, every order is enumerated and priced with the side DP. That is
     exact and cheaper than starting Concorde.
   - Otherwise the **gadget** (§3) is built with `K` taken from the relaxed
     tour's exact time and solved by **Concorde** under `concorde_time_limit`
     (default 1 s).
4. **Validation** (§5). A gadget tour that fails any check is discarded.
5. **Selection.** Every candidate order is re-priced with the **same** exact
   side DP (`_optimize_visit_sides`, float seconds), and the fastest wins. On a
   tie the earlier candidate is kept, in the order relaxed, alignment, exact.
   A failure in step 3 or 4 just drops the exact candidate.

Steps 2–5 can only lower the motion time of a group, never raise it.

## 3. The gadget

### 3.1 Generalized TSP clusters

Each city becomes a **cluster** of ATSP nodes, one per side site. Cluster 0
is the load terminal, clusters `1..C` are the cities, and cluster `C+1` is the
unload terminal.

| Cluster | Nodes (sites) | Leaving cost | Entering cost |
|---|---|---|---|
| load, from home | 4 cardinal offsets | transfer time of that direction | – |
| load, retained | 1 node at `start_offset` | 0 | – |
| city `(r, c)` | `(r±1, c)`, `(r, c±1)` | 0 | 0 |
| unload | 4 cardinal offsets | – | transfer time of that direction |

The cost of an arc `a → b` between different clusters is

```
c(a → b) = depart(a) + route(site(a), site(b)) + arrive(b)      c(unload → load) = 0
```

Choosing one node per cluster and an order gives exactly the exact objective
of §1. This is a generalized TSP (GTSP).

### 3.2 Layer order: one tour across all layers, with no penalty

Clusters carry layers: load is −1, cities have their normalized layer
`0..L−1`, and unload is `L`. An inter-cluster arc `x → y` **exists only if**
`layer(y) − layer(x) ∈ {0, 1}`, plus the single wrap arc unload → load. All
other arcs are absent.

- **No going back.** In a directed tour, every arc except the wrap is
  non-decreasing in layer, and the only arc into the load cluster is the wrap.
  Following the tour from the load cluster therefore never decreases in
  layer: every layer-L city comes before every layer-(L+1) city. The tour
  cannot enter layer L+1 and come back to L, because no such arc exists.
- **Still one joint TSP across layers.** Every pair of same-layer cities, and
  every (layer-L city, layer-(L+1) city) pair, is connected in each allowed
  direction. So every layer-respecting order with every side choice is a tour
  of the gadget. Which city ends layer L and which starts layer L+1 is chosen
  jointly with everything else. That joint choice is what makes a layered
  tour optimal rather than a concatenation of per-layer optima.
- **Why the relaxed ring needs a penalty and the gadget does not.** The
  relaxed graph is undirected, so a cycle can walk an edge in either
  direction, and only a penalty larger than any feasible tour stops it from
  revisiting a layer. In the gadget, direction is enforced by the
  Jonker–Volgenant ties (proved in §4), so leaving the backward arcs out is
  enough. Any tour that tried to walk a forward arc backwards would have to
  traverse a cluster against its ties, which costs at least `K` more than a
  faithful tour.

### 3.3 Noon–Bean (GTSP → asymmetric TSP)

Within each cluster, put the nodes on a directed cycle `σ`
(`v1 → v2 → v3 → v4 → v1`) and:

- **cycle arcs** `a → σ(a)` cost 0;
- **inter-cluster arcs** `a → b` cost `c(σ(a) → b) + K`, the Noon–Bean shift:
  leaving `a` is priced as leaving its cycle successor.

A tour that enters a cluster at `v1` walks `v1 → v2 → v3 → v4` and leaves from
`v4`, priced as leaving from `σ(v4) = v1`. That is "enter at `v1`, leave from
`v1`", exactly the physical constraint. `K` makes splitting a cluster
unprofitable.

### 3.4 Jonker–Volgenant (asymmetric → symmetric)

Concorde solves only symmetric TSPs. Each ATSP node `a` becomes an **in** node
`2a` and an **out** node `2a+1`:

- **tie** `(a_in, a_out)` costs `−K`;
- each ATSP arc `a → b` becomes the edge `(a_out, b_in)` with the arc's cost;
- every other pair is absent.

Every Hamiltonian cycle has the same number of edges, so adding `K` to every
edge does not change which tour is optimal. That gives the non-negative form
the code uses.

### 3.5 Final symmetric weights

| Edge | Weight |
|---|---|
| tie `(a_in, a_out)` | `0` |
| Noon–Bean cycle `(a_out, σ(a)_in)` | `K` |
| inter-cluster `(a_out, b_in)`, allowed by §3.2 | `c(σ(a) → b) + 2K` |
| everything else (absent) | `4K` |

The size is **8 symmetric nodes per city** (4 sites × in/out). Terminals add
16, or 10 for a retained load.

**Why `+2K` and not `+K`.** If intra edges cost `K` and inter edges cost
`route + K`, a cluster's cost becomes `K·(8 − t) + routes`, where `t` is the
number of ties used. The number of separate passes through the cluster drops
out entirely, so nothing stops a tour from leaving a cluster halfway round
and coming back, and the Noon–Bean shifted prices would then undercut the real
optimum. Inter edges need one `K` from Noon–Bean *and* one from
Jonker–Volgenant.

## 4. Correctness

Let `W = Σ_x (m_x + 1)` over clusters `x` of size `m_x`, and let `R(H)` be the
total route cost on the inter-cluster edges of a tour `H`.

**Lemma (half-edge count).** Split each edge's weight equally between its two
endpoints. For any Hamiltonian cycle `H`, the weight charged to the `2m` nodes
of a cluster is at least `K·(m + 1)` plus half the route costs on its inter
edges. Equality holds **only** if `H` passes through the cluster once, in
cycle direction (a faithful Noon–Bean pass). Every other pattern is charged
at least `K` more.

*Proof.* Inside the cluster let `t` be the ties, `i` the cycle edges and `e`
the absent edges that `H` uses, and let `x` be the number of half-edges
leaving the cluster. Degree counting gives `2t + 2i + 2e + x = 4m`. A tie
charges 0, a cycle edge `K/2` per end, an inter edge at least `K` per end, and
an absent edge `2K` per end. The charge is therefore at least
`K·(i + x) + 4K·e ≥ K·(4m − 2t − i)`. The ties and cycle edges `H` uses form a
proper subgraph of the cluster's `2m`-cycle (since `H` covers more than this
cluster), so `t + i ≤ 2m − 1`, and also `t ≤ m`. Hence
`4m − 2t − i ≥ 2m + 1 − t ≥ m + 1`. Equality forces `t = m`, `i = m − 1`,
`e = 0` and `x = 2`, with both external edges present inter edges. That is the
`2m`-cycle minus one cycle edge `(a_out, σ(a)_in)`: a single path entering at
the in-node `σ(a)_in`, running the ties forward, and leaving at `a_out`.
Every quantity besides the route parts is an integer multiple of `K`, so any
deviation costs at least `K`. For `m = 1` (a retained-load terminal) the
cluster is one tie, and `4 − 2t ≥ 2 = m + 1` with equality iff `t = 1`. ∎

**Theorem.** If `K > R*`, the optimal route cost, then every optimal tour of
the gadget is faithful, and it decodes to an optimal visit order *and*
optimal sides.

*Proof.* Summing the lemma over clusters:

- a faithful tour costs exactly `W·K + R(H)`, where `R(H)` is its real motion
  time, because the Noon–Bean prices `c(σ(pred) → next) = c(entry → next)` are
  the physical ones;
- an unfaithful tour costs at least `W·K + K > W·K + R*`.

So the optimum is faithful. Among faithful tours, minimizing `W·K + R` is the
same as minimizing `R`, the exact objective of §1. ∎

**Choosing `K`.** The relaxed tour, re-priced exactly by the side DP, is a
feasible tour, so its time `T_relax ≥ R*`. The code sets `K` just above
`T_relax`, after integer scaling. This is why the relaxed tour is solved
first. A single lattice crossing is **not** enough for `K`: a tour visiting
many cities can cross the lattice many times.

**Absent edges.** `4K` is enough because the lemma already charges absent
edges at `2K` per end. Nothing needs to exceed a whole tour's weight, which
keeps the integer range small (§6).

## 5. Validation of solver output

`decode_gadget_tour` rejects (`GadgetError`) any tour that:

1. is not a Hamiltonian cycle over all `8C + 16` nodes;
2. uses an absent edge (a weight of at least `4K`). This covers backward-layer
   arcs, unload before the last layer, in–in or out–out pairs, and jumps
   inside a cluster;
3. splits an in/out tie, or traverses ties in both orientations, meaning it is
   not a directed tour;
4. visits a cluster more than once (a non-contiguous Noon–Bean pass);
5. passes a cluster's sites against its cycle order;
6. does not run load → cities → unload;
7. decreases in layer.

`_gadget_city_order` then requires every decoded **entry site to be a
cardinal neighbor of its city's target**, the only places a gate can fire.
`_order_group_displacements` checks that the chosen order (relaxed or exact)
is a permutation of the cities with non-decreasing layers. Downstream, the
existing checks still run on the emitted schedule:

- `_execute_group` and the retained-load reroute assert that each gate's
  mover is one grid step from its SLM partner;
- `validate_schedule` replays continuity, highway legality, rigid loads,
  transfers, gate adjacency, Rydberg conflicts, per-qubit gate order and
  barrier epochs.

## 6. Integer scaling for Concorde

Concorde keeps tour lengths and LP bounds in `double`. Its Lin–Kernighan,
however, sums move gains in an `int`, at most `KICK_MAXDEPTH = 50` edges deep
(`LINKERN/linkern.c`). Every edge weight is therefore kept at most
`_CONCORDE_EDGE_CAP = (2³¹−1)/256`. Absent edges only have to outweigh **one
known feasible tour**, not n times the largest edge.

**Relaxed layer ring.** The layer-ordered path is feasible, and its weight
(`_LayeredTourGraph.tour_bound`) sets the scale. All finite weights fit
under the cap, and absent edges weigh that tour plus the n/2 rounding slack.
fast_tsp uses the same rule under its 65535 cap. The older rule (every
n-edge tour fits in int32) left about one integer unit per motion edge at
1000 cities. Concorde then did worse than alignment. Measured on random
single-layer groups, priced exactly against alignment:

| Cities | fast_tsp, old → new | Concorde, old → new |
|---|---|---|
| ~300 | +179% → −22.2% | −22.7% → −22.6% |
| ~1000 | +296% → −26.7% | +31.9% → −30.1% |

**Gadget.** A second Concorde limit applies here. Its exact lower bounds use
fixed-point arithmetic (`CCbigguy`) with about an int32 integer range. When a
tour's weight is too close to 2³¹, Concorde prints `OVERFLOW in
CCbigguy_addmult` and **aborts the process**. A faithful gadget tour weighs
`W·K + R` with `W = Σ(m_x + 1) ≈ 5C`, and `K` has to exceed the whole route
total, so the tour weight grows like `5C · K`. This was measured, not
predicted: a 200-city gadget with `K ≈ 10⁶` (faithful tours around
1.0×10⁹) aborted, and the same matrix divided by 16 solved. The code
therefore takes

```
K_max = min(EDGE_CAP/4, (2³¹−1)/(8·(W+1)), (2³¹−1)/(4n))      n = 8C + 16
units = min(10^6, K_max − C − 2)
scale = units / T_relax
K     = ⌊T_relax · scale⌋ + C + 2,   absent = 4K
```

A faithful tour then stays at most INT_MAX/8, and even a tour of n absent
edges fits in int32. Every weight is also at most `EDGE_CAP`. Rounding moves
each of the `C + 1` route edges of a faithful tour by at most half a unit,
and the `+ C + 2` keeps `K` above the rounded optimum. The final choice always
uses float side-DP prices, so rounding never misreports a time.

The cost is resolution at scale. A route edge gets about `K/C` units:
about 2×10⁴ at 50 cities, ~1,300 at 200 and ~50 at 1,000. That is a
property of the gadget under Concorde's integer range, not of the code. The
relaxed graph does not have this problem, because its whole feasible tour
is scaled to at most `EDGE_CAP`.

**Isolation.** Because Concorde can abort the process, the gadget's Concorde
solve runs in a forked child (`_in_child_process`). An abort, crash or error
there becomes a `_TourSolverError`. The gadget candidate is dropped and
counted in `exact_failures`, and the relaxed or alignment tour is used.

## 7. Configuration and reporting

| Key | Default | Meaning |
|---|---|---|
| `tour_ordering` | `"tsp_gadget"` | `"tsp_gadget"`: relaxed + alignment + exact candidates. `"tsp"`: relaxed (Concorde first) + alignment. `"alignment"`: alignment order only, no TSP. See §2. The old boolean key `tsp_gadget` is rejected with a pointer to this one. |
| `concorde_time_limit` | `1.0` | Seconds per Concorde solve (gadget and relaxed). Soft: see §8. |
| `tsp_time_limit` | `0.05` | Seconds per fast_tsp solve of the relaxed tour. |

`CircuitSchedule.tsp_stats` (a `TourStats`) counts, for the selected
candidate:

- `solves`: groups with at least 2 distinct cities;
- `enumerated` / `gadget`: exact attempts that succeeded, by method;
- `exact_wins` / `alignment_wins` and `seconds_saved`: groups where that
  candidate was strictly fastest, and the motion saved against the relaxed
  order;
- `exact_failures`: exact attempts that fell back to the relaxed tour.

The counters include discarded trial plans (the grouped vs one-atom
partitions, and the reuse options), so they measure solver behaviour rather
than the final schedule. `main` prints one `tsp_…` line. Routine gadget
failures (an unfaithful time-limited tour, too little integer resolution) are
counted silently. Only unexpected errors warn.

## 8. Performance and limitations

Whole-circuit runs (2026-10-17, default limits; "solves" counts every group
with at least 2 cities, trial plans included):

| Circuit | Relaxed only | With exact tours | Change | Exact wins / attempts | Wall time |
|---|---|---|---|---|---|
| BB144 (`bb144_n_init_qldpc`) | 3589.99 µs | 3580.37 µs | −0.27% | 1 / 34 | 1.2 s → 2.2 s |
| QFT-29 (`qft_n29_init`) | 140821.0 µs | 138108.5 µs | −1.93% | 143 / 314 (3 fell back) | 3 s → 176 s |
| VQE-50 (`vqe_n50_init`) | 6260.57 µs | 6260.57 µs | 0 | 0 / 24 (all 2-city) | 0.1 s → 0.1 s |

QFT-29 has groups of up to 54 cities, and most of the extra wall time is
Concorde running up to `concorde_time_limit` on its 199 gadget solves.

QFT-63 (`qft_n63_init_order`, trapezoidal profile, with the FastSA placement
saved once and loaded as `Given`). All three runs price the alignment order
too. With the gadget off, Concorde solves the relaxed tour.

| Setting | Total time | vs gadget off | Exact / alignment wins | Wall time |
|---|---|---|---|---|
| `tour_ordering: "tsp"` | 467296.8 µs | – | – / 34 | 9.8 s |
| gadget, `concorde_time_limit: 1.0` | 464943.6 µs | −0.50% | 299 / 22 (53 exact failures) | 858 s |
| gadget, `concorde_time_limit: 0.2` | 466182.1 µs | −0.24% | 247 / 22 (60 exact failures) | 492 s |

QFT-63 groups never exceed 40 cities. Of the 858 s gadget run, 144 s went to
3–10 city groups, 582 s to 11–20 and 128 s to 21–50.

### Large synthetic groups (2026-10-17)

No benchmark circuit has groups this large, so these are synthetic single
groups. They call the tour code directly, with no circuit, placement or
scheduler involved.

- **Instance:** `random.Random(n)` draws `n` displacements
  `(2·randint(−60, 60), 2·randint(−60, 60))`. Duplicates are dropped, so the
  city count can come out slightly below `n`. One layer, load from home
  (`start_offset = (0, 0)`).
- **Physics:** the test fixture config (8 µm lattice, 10 m/s, 20000 m/s²,
  `Cubic_Spline`).
- **Methods:** within a size, all methods solve the **same** instance. Every
  result is the exact side-DP motion time (`_city_order_seconds`) of the
  returned order. The relaxed solvers use `tour_bound` scaling (§6). The gadget
  uses `K` from the best relaxed tour, and Concorde runs in a child process.
- **Concorde limits:** 120 s at 200 cities, 600 s at 500 and 1800 s at
  1000, for both the relaxed solve and the gadget solve.

| Cities | Alignment | Relaxed fast_tsp (0.05 s) | Relaxed Concorde | Gadget Concorde |
|---|---|---|---|---|
| 200 | 40467.8 µs | 32908.3 µs (−18.7%) | 32821.7 µs (**−18.9%**, <1 s) | 36534.8 µs (−9.7%, 128 s; +11.3% vs relaxed) |
| 496 | 81716.4 µs | 62043.4 µs (−24.1%) | 61559.9 µs (**−24.7%**, 7 s) | 69037.3 µs (−15.5%, 632 s; +12.2% vs relaxed) |
| 969 | 141914.5 µs | 100972.5 µs (−28.9%) | 97590.0 µs (**−31.2%**, 98 s) | 113192.6 µs (−20.2%, 1906 s; +16.0% vs relaxed) |

Build times: the relaxed graph took 4 s, 22 s and 85 s; the gadget took 7 s,
41 s and 146 s (1,616, 3,984 and 7,768 nodes; about 1,300, 217 and 57 units per
route edge). At these sizes the gadget returns valid tours,
but they are clearly worse than the relaxed tour, even with long limits. The
causes are Concorde's weak LP on the gadget and the coarse integer resolution
forced by its bound range (§6). For large groups, use `tour_ordering: "tsp"`:
Concorde on the relaxed graph beats alignment by 19–31% here.

An earlier attempt at 200 cities, before the §6 bound fix, aborted inside
Concorde with `OVERFLOW in CCbigguy_addmult`.

To rerun or extend these measurements, use
[`benchmarks/tsp_group_benchmark.py`](benchmarks/tsp_group_benchmark.py). It
uses the same instance generator and exact side-DP timing. It runs alignment,
fast_tsp, relaxed Concorde and the gadget at several sizes, repeats every
time-limited solver at each `--time-limits` value (quality as a function of
time), prints per-size tables, and writes optional CSV rows. Its docstring
gives usage and rough costs.

- **Concorde's LP bound is weak on Noon–Bean/JV instances.** The root gap is
  roughly 10% of the motion part, so branch-and-bound does the work. Measured
  on random single-layer groups, without a time limit: 9 cities 0.4 s,
  19 cities 5 s, 37 cities not proven within 30 s. More layers help, because
  absent backward arcs prune the search.
- **`concorde_time_limit` is a soft bound.** Concorde checks it only between
  branch-and-bound tasks, so root cutting can overrun it. A capped solve
  returns Concorde's best tour so far. That tour may be suboptimal (the
  comparison then keeps the relaxed tour) or unfaithful (validation rejects
  it). On large groups the gadget therefore acts as an opportunistic improver
  and the relaxed tour carries the result, as intended.
- **"Optimal" is with respect to `parity_route_moves`.** Routes between two
  sites come from that router's candidate set and are taken as fixed. The
  gadget optimizes order and sides, not the router itself.
- **Per-group optimality.** Each load is optimized given the grouping and
  reuse decisions made around it. A faster tour for one group can change
  later greedy choices, so whole-circuit time is not guaranteed monotone,
  although every individual group's motion is.

## 9. Tests

`tests/test_tsp_gadget.py`:

- Concorde on the gadget equals brute force (every layer-respecting order ×
  side DP) on random instances, with 1–3 layers, loads from home and
  retained loads, and both movement shapes.
- A fixed instance where the relaxed tour is 2.65% slower is replaced by the
  exact tour, through both enumeration and the gadget path.
- The weight structure matches §3.5, including that no backward-layer arc
  exists.
- The alignment order competes and wins when it is faster. With the gadget
  off, Concorde solves the relaxed tour first.
- The decoder accepts rotations and reversals and rejects missing or
  repeated nodes, split ties, backward layers, split clusters, and reversed
  cluster passes.
- Unfaithful gadget tours fall back quietly and are counted.
  `concorde_time_limit` reaches Concorde. Configuration validation covers the
  new keys.
