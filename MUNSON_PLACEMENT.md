# Collaborator's initial-placement method (Munson handoff, 2026-10-07)

Source: `~/Quantum_Class/munson-handoff-Oct072026/handoff/`. The method is in
`src/placement.py` and the modules it imports. His own description is
`paper/section_layout.tex`, and his references are in `paper/refs.bib`. This
document explains only the **initial placement**. The handoff also contains
an integrated discovery/layout/route pipeline (`paper/section_layout.tex`
§"Toward an integrated pipeline", parts of `src/discover.py`) and
verification scripts. Those are out of scope here, because our router stays.

References in square brackets, e.g. **[Bentley 1992]**, are explained in §11,
which also maps each one to the step it supports.

## 1. Summary

- **What it is:** a **deterministic minimizer of our own FastSA objective**
  `compute_fastsa_cost` (`src/naive_n_dag/FastSA_Solver.py`). It replaces the
  simulated-annealing search in `run_fastsa`.
- **What it keeps:** the objective F, the layering (including `reorder_cz`),
  the array and the router are all unchanged. It never calls the router.
- **Evaluation budget:** objective evaluations are counted and capped at the
  number FastSA spent on the same circuit (100,103 on his benchmark set).
- **Output:** one layout, the best F found, plus a "pool" of alternative
  layouts. He timed the pool afterwards for analysis; the pool never changes
  the returned layout.

He frames the problem as a quadratic assignment problem (QAP: assigning
objects to sites when the cost depends on distances between assigned sites)
[Koopmans & Beckmann 1957; Burkard et al. 1998]. Standard QAP machinery does
not apply directly, because F divides each layer's cost by the number of
*distinct* displacement vectors in it. F is therefore not a sum of
independent pair terms.

Pipeline (`placement.run`):

```
0. PRESOLVE     cheap structural tests; closed forms (star, path); exhaustive certificate on tiny arrays
1. SEEDS        graph orders x geometric paths, symmetry-derived "torus" layouts, row-major
2. SEARCH       deterministic best-first multistart over the seeds, each descended by a VND
3. POOL         best layout + top seeds + symmetry layouts + objective-identical images (analysis only)
```

Everything is deterministic: there is no random number generator anywhere, and
ties are broken by index.

## 2. Inputs, outputs and coordinates

- **Inputs:**
  - the QASM path;
  - the config (`dimensions` [R, C], `num_NA` = n, `reorder_cz`);
  - FastSA's recorded evaluation count, used as the budget.

  The two-qubit DAG comes from **our** loader,
  `load_qasm_to_two_qubit_dag_with_single_qubit_context(path, reorder_cz=...)`.
  Symmetry discovery separately re-parses the raw QASM text (§5.2).
- **Layout representation:** a list indexed by qubit, holding `(row, col)` SLM
  **site indices** in `0..R-1 × 0..C-1`. The objective sees grid coordinates
  `x = 2·col`, `y = 2·row`, the same even-even convention as `grid.py`.
- **Delivery:** `layouts/<circuit>.layout.json` holds `[row, col]` per qubit.
  That's the exact format our `fill_strategy: "Given"` and
  `initial_layout_fill` read (which doubles the indices). His delivered
  configs differ from ours only in `fill_strategy` and `initial_layout`
  (`deliver/README.md`).

## 3. The objective and how it is evaluated

F is our `compute_fastsa_cost`. For each two-qubit layer ℓ with gate
displacement vectors v_g:

```
c_ℓ = (1/|D_ℓ|) · Σ_{d ∈ D_ℓ} |d| / count(d)        D_ℓ = distinct canonical vectors (v ~ −v)
a_ℓ = mean over gate pairs of |cos θ| · min(|u|,|v|)/max(|u|,|v|)
F   = (1/L) · Σ_ℓ (c_ℓ − a_ℓ)
```

**Vectorized evaluator (`src/fastcost2.py`).** Every search step calls
`compute_cost_fast`, not our Python version. It relies on two exact
simplifications:

1. The layering and the gate index pairs are fixed during the search, so
   `CompiledCircuit` precomputes them once: per-gate `q0`, `q1` and layer,
   plus every within-layer gate pair (`pi`, `pj`).
2. `|cos θ| · min/max = |u·v| / max(|u|², |v|²)`. This removes every square
   root from the O(m²) alignment term. Per-layer sums use `np.add.reduceat`
   over the layer-sorted arrays.

Distinct canonical vectors are counted per layer with one sort-and-run-length
pass. He reports this as bit-identical to our formula at commit `465a322`.

**Invariances he exploits (presolve rule P2).** F depends on the layout only
through displacement vectors. It is therefore unchanged by reflecting the
layout and by translating it inside the array, and on square arrays by
rotations too. That gives the D4 group, or the Klein four-group
(the four reflections and rotations that keep a non-square grid) when R ≠ C.
He uses this to skip equivalent layouts and to generate "postsolve" images.

> **Drift since his commit (checked 2026-10-17).** Our `compute_fastsa_cost`
> changed after `465a322`. It now drops barrier and empty layers and divides
> by the number of non-empty gate layers. His `fastcost2` still divides by
> all DAG layers. The two differ by the constant factor (non-empty layers /
> all layers): 12/14 on `bb_n72`, 8/10 on `surface_rotated_n25` and
> `toric_n18`, and 110/111 on `qft_n29`. Ranking, and so the minimizer, is
> unaffected, but the absolute F values differ. His "agrees exactly with
> `compute_fastsa_cost`" check would now fail by that factor.

## 4. Stage 0 — Presolve (`src/presolve.py`, `src/solve.py`)

He structures this like LP/MIP presolve, with ASAP layers as rows, qubits as
columns and gates as nonzeros. Each rule is a cheap O(G) test on the gate
list and never evaluates F.

| Rule | Test | Consequence |
|---|---|---|
| P1 idle | qubits in no two-qubit gate | They don't affect F; fill them into leftover sites last. |
| P2 invariance | always | Evaluate one layout per reflection/rotation/translation class. |
| P3 singleton layers | every layer has one gate | Alignment is 0, and F = (1/L) Σ ‖v_g‖. |
| P4 star | P3 and every gate shares one hub qubit | **Closed form**, globally optimal. |
| P5 path | P3 and the interaction graph is a path | **Serpentine construction** meets the lower bound F = 2. |
| P6 uniform | every pair interacts with the same multiplicity | Only *which* sites are occupied matters (informational). |
| P7 layer law | every multi-gate layer has constant i+j, or constant j−i | Affine/serpentine placements make each layer parallel. |
| P8 automorphisms | layer-preserving automorphism count, by propagation | Symmetry-reduced route if > 1. |

The dispatch picks one route, in this order: P4 → P5 → P8 → P7 → GENERIC. In
`placement.run` the route only (a) short-circuits P4/P5 to their closed form
and (b) reorders seed groups for P7/P8.

**P4 star closed form (`_star_closed_form`).** With one gate per layer and a
common hub, F separates over leaves. For every candidate hub site, the leaves
take the nearest other sites. The best hub site is a proved global optimum
in effectively one evaluation.

**P5 path construction (`_path_closed_form`).** The path is walked from an
endpoint and laid along a serpentine of width C, so consecutive qubits are
adjacent and every gate has length 2. Two is the minimum distance between
distinct even-even sites, so this meets the lower bound.

**Closed-form pool.** F ignores idle qubits and where the active block sits.
So the closed-form route also emits up to 64 objective-identical variants:
D4 images and translations of the *active* qubits' bounding box, with idle
qubits refilled around them. These cost zero evaluations.

**Exhaustive certificate (`exhaustive_certificate`, "P6" route).** This is
tried only if `perm(R·C, n) ≤ budget × |G|`, where G is the grid isometry
group (order 8 if square, else 4). He calls this "an exact necessary
condition, not a tuned threshold". When it holds, every injection of n
qubits into the R·C sites is enumerated, and only one canonical
representative per isometry orbit is scored. Example from his comments:
`qft_7` takes 22,680 evaluations, against FastSA's 100,728. The result is a
proved global optimum.

On the 14 qLDPC circuits **none of these fired**: every circuit took the
GENERIC route (`out/results.json`).

## 5. Stage 1 — Seeds (`placement.seed_pool`)

Every distinct seed is scored once and the list is sorted by F. There are
three families.

### 5.1 General structured seeds (`src/general_seed.py`, `seed_candidates_v2`)

These are built from the weighted interaction graph W, where W[i, j] is the
number of two-qubit gates between i and j. Each one is a pair of a **linear
order** and a **geometric path**:

- **Orders** (minimum-linear-arrangement heuristics, which put strongly
  interacting qubits close together on a line):
  - `identity`;
  - `spectral`: the Fiedler ordering, i.e. sort by the eigenvector of the
    second-smallest Laplacian eigenvalue [Fiedler 1973];
  - `rcm`: reverse Cuthill–McKee, a bandwidth-reducing BFS ordering
    [Cuthill & McKee 1969];
  - `degree`: highest weighted degree first.
- **Paths through the R×C sites:**
  - serpentine and raster paths of every width w = 1..max(R, C), along
    either axis;
  - radial paths: sites sorted by L1 or L2 distance from each centre in one
    quadrant (the other quadrants follow by symmetry). These suit star-like
    graphs.

Every order × path combination is generated and duplicates are dropped. He
estimates the whole seed pool at about 400 evaluations.

### 5.2 Symmetry-derived "torus" (equivariant) seeds (`src/discover.py`, `src/schedule_compare.py`)

This targets circuits with a translation symmetry, such as bivariate-bicycle
codes. The idea is that if the circuit is invariant under a group
H ≅ Z_L × Z_M acting freely on the qubits, and H is realised as grid
translations, then every gate in a symmetry orbit gets the **same**
displacement vector. That is exactly what F's distinct-vector term rewards.

1. **Parse** (`Circuit`). The raw QASM text is re-parsed, *not* the reordered
   DAG. Edges are labelled by **block**: a maximal run of consecutive
   two-qubit gates, which for syndrome circuits means one check round. Header
   comments are ignored on purpose ("header-blind").
2. **Colour refinement.** The initial colour of each qubit is its degree in
   each block (`block_degree_signature`). This is refined with
   1-dimensional Weisfeiler–Leman on the block-labelled multigraph
   (`refine`) [Weisfeiler–Leman; see §11].
3. **Translation search (`find_translations`).** This is
   individualisation–refinement [McKay & Piperno 2014]. For each candidate
   image y of a root qubit x with the same colour, it tags x and y, re-refines
   both colourings, and branches on the smallest non-singleton class until the
   map is determined. A map is kept only if it is **fixed-point-free** and a
   true automorphism. The search is capped by search nodes (20,000) and by
   refinement work (50M operations / |E| rounds). A capped search is
   recorded, so it isn't mistaken for "no symmetry".
4. **Free abelian subgroup (`free_abelian_subgroup`).** It looks for the
   largest group generated by one or two commuting discovered maps in which
   **every** non-identity element is fixed-point-free. Generators are ranked
   by the size of the free cyclic group they generate, not by order, because
   the highest-order maps are often a translation composed with a
   reflection.
5. **Presentations (`presentations`).** These are all ways to write
   H = ⟨g1⟩ × ⟨g2⟩ with |g1|·|g2| = |H|. The same group can have several
   physical shapes: Z_18 × Z_3 and Z_9 × Z_6 give 36×6 and 18×12 footprints.
6. **Construction (`equivariant_variants`, `_variant`).** For each
   presentation (L, M):
   - group element g1^i·g2^j becomes torus point (i, j);
   - each of the at most 4 orbits gets one cell (dr, dc) of a 2×2 block;
   - qubit h(rep_orbit) goes to site (2i + dr, 2j + dc).

   It enumerates all 4! orbit→cell assignments × 2 axis orders. The variant
   is kept only if 2L ≤ R and 2M ≤ C. Qubits outside the circuit's support
   take spare sites in row-major order, so the layout stays injective.

In his comments, the 2L×2M footprint fits the shipped array on only 2 of the 6
bivariate-bicycle codes (`bb_n72`, `bb_n288`). A generalisation, subgroup
tilings with p×q cells and shear (`src/quotient.py`), exists but is **off for
seeding** (`USE_QUOTIENT_SEEDS=False`). It gave a better seed on `bb_n360` but
a 17.2% worse final F, because equivariant layouts are deep local optima for
swap moves. It is used only to diversify the pool.

### 5.3 Row-major

`[(i // C, i % C)]`.

### 5.4 Ranking

- The groups are concatenated (`rowmaj, general, torus` for GENERIC; torus
  first for P8; general first for P7), de-duplicated, scored once each, and
  sorted by F, with ties going to the earlier entry.
- The best torus seed is remembered separately ("protected") so it reaches
  the pool even if it ranks poorly on F.
- An optional "antiparallel screen" re-ranking (`SCREEN_PENALTY`) is **off**.

## 6. Stage 2 — Search

### 6.1 Deterministic multistart (`src/heap_det.py`)

Seeds go into a min-heap keyed by F, and the loop pops them best-first:

- **Diversity veto:** a popped seed is skipped if it differs, in fewer than
  `max(2, n//6)` qubit positions, from any seed or descent **endpoint**
  already descended (`ENDPOINT_VETO=True`).
- Each surviving seed is descended with the VND (§6.2), and the best result
  is kept.
- **Stop rules:**
  - the heap is exhausted;
  - a **stall**: one descent without improving the best, but only after 10%
    of the budget has been spent (`STALL=1`, `MIN_SPEND_FRAC=0.10`);
  - the **budget cap**: `budget − 122` evaluations are reserved for the pool
    stage.
- **Runner-up re-seeding:** pushing a descent's best rejected neighbours back
  onto the heap is implemented but **off** (`PUSH_PER_DESCENT=0`). He found
  identical results with it on.

In his runs this meant 1–8 descents per circuit. Five of the six
bivariate-bicycle codes hit the budget cap after 1–2 descents.

### 6.2 Variable neighbourhood descent (`src/vnd.py`)

He adapts VND [Mladenović & Hansen 1997]. The neighbourhoods run in a fixed
order:

```
swap(k=12) → layer(12) → swap(24) → layer(24) → swap(48) → layer(48)   [→ ejection chains: off]
```

After any improvement the sequence restarts at `swap(12)`; otherwise it
moves to the next neighbourhood. It stops when the last neighbourhood fails
or the budget runs out. The first call is never treated as an improvement,
which avoids a wasted restart.

### 6.3 Swap/relocate descent with candidate lists (`src/descent_fast.py`)

This uses the standard QAP neighbourhood (swap two qubits, or move one qubit
to a free site), restricted so that one pass costs O(n·k) rather than
O(n²). One pass at n = 720 drops from about 265k evaluations to about 14k.

- **Candidate partners for qubit i** (`candidate_lists`), computed once at the
  start of each descent from the positions at that time:
  - i's interaction-graph neighbours;
  - the k nearest qubits by current position, with k = 12, 24 or 48 from the
    VND phase;
  - 6 "well-spread" qubits from i's distance-sorted list, at golden-ratio
    strides offset by i. This is a deterministic low-discrepancy sample that
    gives every qubit a few distant partners.
- **Moves tried for i:**
  - a swap with each candidate;
  - a relocation to each of the 4 nearest free sites.

  It takes the best improving move among these (best improvement;
  `FIRST_IMPROVE=False`).
- **Don't-look queue** [Bentley 1992]. All qubits start in a FIFO queue.
  After an accepted move, only the moved qubits **and their interaction
  neighbours** are re-queued (`WAKE_ADJ=True`). Bentley's original rule wakes
  only the endpoints. He notes that rule is exact for the TSP, where cost is
  a sum of edge costs, but not for F: |D_ℓ| and the alignment term are
  layer-global, so the neighbour wake is a cost/quality compromise. The
  descent ends when the queue is empty.
- This certifies a local optimum of the **restricted** neighbourhood only.
  That's weaker than a full O(n²) sweep (`src/descent.py`, which `heap_det`
  falls back to when no descent function is given).

### 6.4 Layer-guided moves (`src/neigh_layer.py`)

This neighbourhood targets the term F actually charges, |D_ℓ|. For every
layer with at least 2 gates and more than one distinct canonical vector:

- find the **majority** displacement m;
- for every gate (a, b) whose displacement differs, try four moves: move b
  to a ± m, or move a to b ± m;
- if the target site is occupied, swap with its occupant; skip targets
  outside the array.

Per round, the best improving move over all candidates is accepted, for up
to 200 rounds. That's about 2,160 moves on `bb_n360`, against 265k swap
pairs.

### 6.5 Ejection chains (`src/neigh_chain.py`) — disabled

These are cyclic chains of displacements (q1 → hole, q2 → q1's site, …) up to
depth 5 and width 6 [Rego, James & Glover 2010; see §11]. They're off
(`USE_CHAIN=False`) because his ablation showed no measured gain.

## 7. Stage 3 — Pool and postsolve (analysis only)

The returned layout is always the search's best F (`chosen_by="objective"`).
Separately, a pool of at most 32 layouts is assembled for afterwards timing
with the router:

- the search best;
- the 8 best seeds;
- up to 8 subgroup-equivariant layouts (`quotient.py`);
- up to 10 torus layouts, plus the protected torus seed;
- 8 **postsolve** images: D4 images and shifts of up to 1 of the best layout,
  all with **identical** F.

Each member also gets an "antiparallel screen" score, `src/cost_signed.py`,
which is descriptive only. The pool keeps half its slots for the best by F
and half for torus/equivariant layouts, which tend to rank badly on F. Pool
scoring is counted inside the budget.

## 8. Budget accounting

- Every call to `compute_cost_fast` counts as one evaluation. That covers
  seeds, every descent trial, pool scoring and screens.
- The cap is FastSA's recorded evaluation count for that circuit; the
  default is 100,103.
- The cap is enforced **inside** descents, per trial. Without that,
  `bb_n288` reached 136,420.
- The final layout is re-checked once with the original
  `compute_fastsa_cost`, reported as `verify_delta`.

## 9. Switches (top of `placement.py`)

| Flag | Value | Meaning |
|---|---|---|
| `BUDGET` | 100,103 | Default cap if no FastSA count is given. |
| `K_LADDER` | (12, 24, 48) | Nearest-neighbour candidate sizes per VND phase. |
| `K_DISC` | 6 | Golden-ratio "distant" candidates per qubit. |
| `USE_LAYER` | True | Layer-guided neighbourhood in the VND. |
| `USE_CHAIN` | False | Ejection chains (no gain). |
| `FIRST_IMPROVE` | False | Best improvement within each qubit's list. |
| `WAKE_ADJ` | True | Re-queue interaction neighbours after a move. |
| `MIN_SEP` | `max(2, n//6)` | Diversity veto distance (qubits in different sites). |
| `ENDPOINT_VETO` | True | Veto against descent endpoints as well as seeds. |
| `STALL`, `MIN_SPEND_FRAC` | 1, 0.10 | Stop after one non-improving descent, once 10% of the budget is spent. |
| `PUSH_PER_DESCENT` | 0 | Runner-up re-seeding (off). |
| `USE_QUOTIENT_SEEDS` | False | Subgroup-tiling seeds (off; pool only). |
| `SCREEN_PENALTY` | False | Antiparallel re-ranking of seeds (off). |
| `POOL_MAX`, `PS_TOP`, `POOL_SEEDS`, `TORUS_POOL`, `QUOT_CAP` | 32, 8, 8, 10, 8 | Pool composition. |

## 10. His results (`paper/table_layout.tex`, `out/results.json`)

These are the 14 qLDPC syndrome circuits with his configs, compared with our
recorded FastSA seed-0 runs at `465a322`. The router, layering and objective
are the same. F values use the old normalization (§3).

| Circuit | F FastSA | F his | Evals (cap 100,103) | Stop | Texec FastSA (ms) | Texec his (ms) | Texec reduction |
|---|---|---|---|---|---|---|---|
| bb_n72 | 8.4231 | 4.6886 | 75,580 | stall | 100.85 | 54.68 | 45.8% |
| bb_n90 | 10.0228 | 7.4222 | 99,997 | cap | 131.40 | 107.17 | 18.4% |
| bb_n108 | 11.6228 | 6.8465 | 100,021 | cap | 167.31 | 108.70 | 35.0% |
| bb_n144 | 12.9670 | 7.5537 | 100,005 | cap | 213.90 | 144.55 | 32.4% |
| bb_n288 | 19.4732 | 5.4936 | 100,036 | cap | 455.17 | 65.70 | 85.6% |
| bb_n360 | 22.7133 | 11.6059 | 100,021 | cap | 591.84 | 204.94 | 65.4% |
| hgp_hamming_n58 | 5.4357 | 3.3547 | 67,422 | stall | 59.37 | 43.45 | 26.8% |
| hgp_repetition_n13 | 1.8257 | 0.3864 | 11,992 | stall | 8.87 | 3.80 | 57.2% |
| hgp_repetition_n41 | 4.6879 | −0.2437 | 18,424 | stall | 29.54 | 4.34 | 85.3% |
| surface_rotated_n9 | 1.2301 | 1.1147 | 12,193 | stall | 4.91 | 4.59 | 6.5% |
| surface_rotated_n25 | 2.5187 | 1.6214 | 22,688 | stall | 14.94 | 11.69 | 21.8% |
| surface_rotated_n49 | 5.4553 | 2.9007 | 41,828 | stall | 38.29 | 23.33 | 39.1% |
| toric_n18 | 2.4387 | 1.6704 | 12,526 | stall | 14.29 | 10.12 | 29.2% |
| toric_n50 | 5.5021 | 2.8400 | 33,555 | stall | 39.67 | 18.91 | 52.3% |

- **Overall:** F is lower on all 14 circuits (median −43.0%), and so is Texec
  (median −37.1%).
- **Budget:** five of the six bivariate-bicycle codes hit the cap, so those
  layouts are not converged.
- **Search vs seeds:** the search improves F over the best seed by up to
  58.9%. On `bb_n288`, both `hgp_repetition` codes and `toric_n18`, the best
  seed (a torus or Cuthill–McKee layout) was already a local optimum.
- **Where lower F ran slower** (his "Limitations of the objective"):
  - On `bb_n360` and `bb_n90` his layouts beat row-major on F yet *run
    slower* than row-major (204.94 vs 153.70 ms, and 107.17 vs 84.04 ms).
  - Across his timed layouts, 22.3% of pairs (833 of 3,729) are ranked
    oppositely by F and by Texec.
  - F ignores motion **direction**: |cos θ|, and v ~ −v in D_ℓ.
  - Layouts with identical F (reflections, translations) differ in Texec by
    up to 4.0%.

  So better minimisation of F improves Texec only where F tracks Texec. This
  bears directly on our task of improving the FastSA cost function.

## 11. Sources: which paper each idea comes from

His `paper/refs.bib` has 12 entries. Seven support the placement method. The
rest support his routing/layering pipeline, not placement. Some techniques
are named only in code comments without a bibliography entry; those are
marked "not in his bib".

| Technique in the placement method | Where | Source |
|---|---|---|
| Placement as a quadratic assignment problem | §1; paper "placement objective" | Koopmans & Beckmann, "Assignment Problems and the Location of Economic Activities", *Econometrica* 25(1), 1957, doi:10.2307/1907742. Burkard, Çela, Pardalos & Pitsoulis, "The Quadratic Assignment Problem", TU Graz tech. report, 1998. |
| Swap / relocate neighbourhood | §6.3 (`descent.py`, `descent_fast.py`) | The standard QAP neighbourhood (`descent.py` calls it "the standard QAP one"), covered by Burkard et al. 1998. No separate citation. |
| Spectral (Fiedler) ordering | §5.1 (`order_spectral`) | Fiedler, "Algebraic Connectivity of Graphs", *Czechoslovak Math. J.* 23(2), 1973, doi:10.21136/CMJ.1973.101168. |
| Reverse Cuthill–McKee ordering | §5.1 (`order_cuthill_mckee`) | Cuthill & McKee, "Reducing the Bandwidth of Sparse Symmetric Matrices", *Proc. ACM National Conf.*, 1969, doi:10.1145/800195.805928. (The *reversal* is usually credited to George, 1971; not in his bib.) |
| Individualisation–refinement symmetry search | §5.2 step 3 (`find_translations`) | McKay & Piperno, "Practical Graph Isomorphism, II", *J. Symbolic Computation* 60, 2014, doi:10.1016/j.jsc.2013.09.003. |
| 1-dim Weisfeiler–Leman colour refinement | §5.2 step 2 (`wl_refine`, `refine`) | Named in code, **not in his bib**. Usually credited to Weisfeiler & Leman (1968); the paper text folds it under McKay & Piperno 2014. |
| Variable neighbourhood descent (restart on improvement) | §6.2 (`vnd.py`) | Mladenović & Hansen, "Variable Neighborhood Search", *Computers & OR* 24(11), 1997, doi:10.1016/S0305-0548(97)00031-2. The first- vs best-improvement trade-off in `descent_fast.py` is attributed to "Hansen & Mladenovic" in a comment. |
| Don't-look bits, queue form | §6.3 | Bentley, "Fast Algorithms for Geometric Traveling Salesman Problems", *ORSA J. Computing* 4(4), 1992, doi:10.1287/ijoc.4.4.387. He adapts the wake rule (neighbours as well as endpoints). |
| Ejection chains (disabled) | §6.5 (`neigh_chain.py`) | Cited in code as "Rego, James & Glover (2010)", **not in his bib**. Presumably Rego, James & Glover, "An ejection chain algorithm for the quadratic assignment problem", *Networks*, 2010; verify before citing. |
| Golden-ratio low-discrepancy candidate sampling | §6.3 (`candidate_lists`) | Described in code as "the standard low-discrepancy choice in one dimension". **No citation.** |
| Presolve framed like LP/MIP presolve; orbit/symmetry fixing; exhaustive search over isometry classes; star/path closed forms | §4 | His own constructions. **No citation.** |
| Layered quantities used only by his pipeline: Kőnig edge colouring, Kempe chains, Vizing via Misra–Gries | `discover.py` (not used by placement) | Kőnig, *Math. Annalen* 77, 1916, doi:10.1007/BF01456961. Misra & Gries, "A Constructive Proof of Vizing's Theorem", *IPL* 41(3), 1992, doi:10.1016/0020-0190(92)90041-S. Kempe chains: **not in his bib**. |
| Routing in his integrated pipeline (not placement) | paper §"Route" | Knuth & Plass, "Breaking Paragraphs into Lines", 1981, doi:10.1002/spe.4380111102. Held & Karp, "A Dynamic Programming Approach to Sequencing Problems", 1962, doi:10.1137/0110015. Croes, "A Method for Solving Traveling-Salesman Problems", 1958, doi:10.1287/opre.6.6.791. |

## 12. Implementation in this repo: `fill_strategy: "deterministic"`

The method is ported into one new module,
[`src/naive_n_dag/deterministic_placement.py`](src/naive_n_dag/deterministic_placement.py).
It is selected with `fill_strategy: "deterministic"`. No existing strategy is
changed: `main.py` gains one dispatch branch and validation of the options
block. Tests are in `tests/test_deterministic_placement.py`.

**Objective (since 2026-10-20): F_time, not F.** His method minimises F
(§3). Our port minimises the same cost FastSA now uses, F_time, the
time-based estimate in COST_FUNCTIONS.md. F was removed from `src/` after the
comparison there showed faster schedules with F_time. Everything below that
describes *his* code (§1–§11) still refers to F.

```json
"fill_strategy": "deterministic",
"Deterministic_fill": {"budget": 100103}
```

| `Deterministic_fill` option | Default | Meaning (§ above) |
|---|---|---|
| `budget` | 100,103 | Hard cap on objective evaluations; a run never exceeds it. |
| `k_ladder` | [12, 24, 48] | Nearest-qubit candidate sizes, one swap phase and one layer phase each (§6.2–6.3). |
| `k_disc` | 6 | Golden-ratio "distant" candidates per qubit (§6.3). |
| `free_per_qubit` | 4 | Free sites tried per relocation (§6.3). |
| `layer_moves` | true | Layer-guided neighbourhood (§6.4). |
| `symmetry_seeds` | true | Torus seeds from the discovered translation group (§5.2). |
| `closed_forms` | true | Star / path closed forms (§4). |
| `min_sep` | null → `max(2, n//6)` | Diversity veto (§6.1). |
| `stall`, `min_spend_fraction` | 1, 0.10 | Stall rule (§6.1). |

`main` prints one line per run:
`deterministic_placement route=… objective=… evaluations=…/… descents=… stop=…`.
The Python API is `run_deterministic_placement(dag, n, (R, C),
reference_nodes=…, options=…, config=…)`; `config` supplies F_time's
physics, and `cost=` substitutes another cost (the benchmarks pass the legacy
F from `benchmarks/legacy_f.py`). It returns a
`DeterministicPlacementResult` (layout, objective, evaluations, route, stop
reason, best seed, descents, symmetry group order). There is also
`deterministic_fill(grid, n, dag, config, …)`, mirroring `Fastsa_Fill`.

**Differences from his code**

1. **Objective.** F_time instead of F (see above). The star closed form
   ranks sites by round-trip move time. The path construction still puts
   every gate on adjacent sites.
2. **Same DAG as `fastsa`.** Placement sees the two-qubit DAG `main`
   passes to FastSA. In the current pipeline that is the *un-reordered* DAG,
   because `reorder_cz` runs after placement and uses it. His results
   (§10) were on the reordered layering of the older pipeline, so the
   numbers are not directly comparable.
3. **Symmetry discovery input.** Blocks come from `reference_nodes`, the
   program-order op list `main` already loads, rather than from re-parsing
   the QASM text. It therefore also works when `step_order` replaces the
   QASM file; there the two-qubit DAG is used as a single block. Barriers
   close a block and are never edges.
4. **Closed forms, made exact.**
   - Star: leaves that interact with the hub more than once are weighted
     (most frequent leaf to the nearest site), where his code overwrote
     them.
   - Path: the path rule requires a true path (connected, acyclic); his
     accepted cycles, for which the serpentine is not optimal.
5. **Dropped.** The analysis pool and postsolve images (§7), the
   antiparallel screen, and features he had switched off (ejection chains,
   subgroup-tiling seeds, runner-up re-seeding, screen re-ranking). The
   presolve tags P1–P3 and P6–P8 are informational in his code; only P4/P5
   and the seed ranking are kept. The exhaustive isometry-class route was
   kept at first and removed with F, because its proof relies on F's exact
   invariance under reflections. Seed groups are always generated as
   row-major, general, torus (ties go to the earlier one).
6. **Budget.** The cap applies to the whole run. The stall rule's 10%
   minimum spend uses this run's budget; he used the module constant. Each
   descent checks the cap before every trial, so the count never exceeds
   `budget`.
7. **Determinism.** Every sort is stable, including the RCM and degree
   orders, so results do not depend on numpy's sort algorithm.

**Measured in the current pipeline (2026-10-20, both placers still
minimising F, FastSA with its settings at the time).** Each run is `main` on
the qLDPC configs as shipped (`reorder_cz: true`, `T_reuse: 3`,
`prefer_moving_ancillas: true`). Only `fill_strategy` changes: `fastsa`
uses its `Fastsa_fill` block and seed 0; `deterministic` uses the default
budget. Texec is the schedule's `estimated_time`; wall time is the whole
`main` call.

| Circuit | Texec `fastsa` | Texec `deterministic` | Change | Wall time | Evaluations used / stop |
|---|---|---|---|---|---|
| surface_rotated_n9 | 1518.9 µs | 1492.7 µs | −1.7% | 54 s → 0.7 s | 10,732 / stall |
| hgp_repetition_n13 | 2219.2 µs | 689.1 µs | −68.9% | 71 s → 0.7 s | 10,919 / stall |
| toric_n18 | 3874.2 µs | 1726.1 µs | −55.4% | 105 s → 3.3 s | 12,018 / stall |
| surface_rotated_n25 | 5445.4 µs | 4415.9 µs | −18.9% | 142 s → 3.9 s | 14,967 / stall |
| hgp_repetition_n41 | 8959.6 µs | 689.1 µs | −92.3% | 218 s → 1.6 s | 16,643 / stall |
| surface_rotated_n49 | 11241.7 µs | 8689.3 µs | −22.7% | 300 s → 11 s | 51,016 / stall |
| toric_n50 | 12339.8 µs | 5320.3 µs | −56.9% | 287 s → 9.4 s | 22,848 / stall |
| bb_n72 | 21848.0 µs | 8332.8 µs | −61.9% | 560 s → 29 s | 40,926 / stall |

All eight runs stopped by the stall rule, well under budget. The larger
bivariate-bicycle codes were not run here, because FastSA alone takes many
minutes on them. FastSA is a single seed (seed 0). Its F was not logged in
these runs, so this table does not by itself show whether lower F caused
the lower Texec.

The collection runner `qLDPC_Qasms/run_naive_n_dag.py` still accepts only
`fastsa` and `random`. Use `main` directly, or extend that runner, to batch
the new strategy.
