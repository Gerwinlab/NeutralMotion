# Placement cost: F_time

Implementation: [`src/naive_n_dag/placement_cost.py`](src/naive_n_dag/placement_cost.py).
Tests: [`tests/test_placement_cost.py`](tests/test_placement_cost.py).

Both placers, `fill_strategy: "fastsa"` (simulated annealing,
`FastSA_Solver.run_fastsa`) and `fill_strategy: "deterministic"`
(`deterministic_placement.py`, see MUNSON_PLACEMENT.md), minimise **F_time**:
an estimate, in µs, of the two-qubit execution time that depends on the
placement (§2). The router is never called inside either placer.

**History.** Until 2026-10-20 the placement objective was the original FastSA
cost F (`compute_fastsa_cost`). That day both costs were compared on general
circuits (§3–§5). F_time gave faster schedules with both placers once
FastSA's schedule was normalised and tuned, so it replaced F, and F was
removed from `src/`. Specifically:

- the `cost_function` config key was removed; `main` rejects it with a
  pointer here;
- the deterministic placer's exhaustive isometry-class route was removed. It
  relied on F being exactly invariant under reflections, which F_time is not
  (§2).

A copy of F is kept in [`benchmarks/legacy_f.py`](benchmarks/legacy_f.py):
the original `compute_fastsa_cost` and a vectorised `FastSACost`. Both placers
accept a substitute cost through `cost=`, so `benchmarks/cost_fidelity.py`
and `benchmarks/fastsa_tuning.py` can still reproduce the comparison.

## 1. The legacy objective F (reference)

F was the objective FastSA annealed:

```
per layer ℓ:  cost_ℓ  = (1/|D_ℓ|) Σ_{distinct ±v} |v| / count(v)
              align_ℓ = mean over gate pairs of |cos θ| · min/max(|u|, |v|)
F = (1/L) Σ_ℓ (cost_ℓ − align_ℓ)
```

The intent is to maximise parallel gates (one displacement serves many
gates) and to keep travel short, using alignment as the measure of how close
vectors are, as the alignment ordering and the grouping rule use it. F_time
keeps that intent but measures it in time (§2).

The original `compute_fastsa_cost` rebuilt the DAG's layers on every call.
The vectorised copy computes the same values 8–119× faster (5.5 ms → 0.1 ms
per evaluation on QFT-29). F_time costs about what the original F did:
between 0.1× and 1.7× of it, with a median of about 0.8×, across seven
benchmark circuits.

## 2. F_time

F_time measures directly what F approximates, in µs, using the router's own
rules and motion law (`dynamics._step_seconds`):

- **Parallel gates** become *stops*: a set of gates with one displacement is
  moved together and fired in one Rydberg pulse.
- **Travel** is the hop *between* stops within a load, timed with the
  router's motion profile. This replaces "alignment = close". A pair u, −u
  is aligned under |cos θ| but is the longest hop when the mover is fixed.
- **Grouping** decides how many loads a layer needs, and every load pays two
  SLM↔AOD transfers. This is where alignment really matters: with
  `alignment_conc` > −1, poorly aligned gates are forced into extra loads.

```
F_time = Σ_units [ loads · (2·t_transfer + 2·t_step)  +  stops · (t_2q + t_switch)  +  tour time ]
```

Time that does not depend on the placement is left out: single-qubit
layers, and the gate count itself. So only *differences* between layouts are
meaningful, and F_time is not a prediction of Texec. On QFT-29's given
layout it reads 173.7 ms against a scheduled 140.8 ms. It runs high because
it ignores load reuse and uses a nearest-neighbour tour instead of the
scheduler's TSP.

### Two modes, matching how the scheduler forms its layers

**Fixed layers** (`reorder_cz` off; e.g. QFT). This mode runs per DAG layer:

1. **Movers.** The ancilla moves when `prefer_moving_ancillas` is set and the
   gate has exactly one ancilla (a qubit that is reset). Otherwise either
   operand may move.
2. **Loads**, emulating `dynamics._build_groups`. Each gate joins the first
   load whose movers still fit the AOD span (`max_dimension`) and, unless
   `alignment_conc` is −1, whose vectors it aligns with at least
   `alignment_conc` (signed cosine, as the router uses). It takes its
   best-aligned mover choice; otherwise it opens a new load.
3. **Stops per load:** distinct displacements. They are signed when the mover
   is fixed. With a free mover, v and −v count as one stop, which is
   optimistic but reflects that the router can move either operand.
4. **Tour per load:** a nearest-neighbour tour from home through the stops
   and back. Each hop is timed with the router's motion law, in grid steps of
   `lattice_spacing / 2`.

**CZ sections** (`reorder_cz` on). Here the scheduler re-layers each commuting
CZ section *after* placement (`dag_helper.reorder_cz_layers_by_displacement`):
gates are bucketed by signed displacement, each bucket is split into
matchings, and the matchings are packed into layers. F_time models that per
section (`dag_helper._commuting_cz_sections`):

1. **Movers** use the reorder's rule: the ancilla when preferred, else the
   first operand. Displacements are therefore signed.
2. **Stops:** each displacement bucket needs as many stops as its largest
   qubit degree, i.e. its number of matchings. That count is exact for
   bipartite data–ancilla rounds.
3. **Loads:** the section needs at least its largest qubit degree in layers,
   one load each. This term does not depend on placement.
4. **Tour:** one tour visits the section's distinct stops, as a load retained
   across the section's layers can.
5. **Other gates:** non-CZ two-qubit gates are single-gate layers, priced as
   in fixed mode.

### Not modelled (possible refinements)

- parity-highway detours and the one-step side offset at each stop;
- Rydberg-radius pulse splitting of nearby simultaneous gates;
- load reuse and held atoms across units, beyond the section tour;
- in section mode, the alignment and AOD-span grouping;
- the scheduler's tour order: TSP, gadget or alignment, rather than
  nearest-neighbour.

### Interaction with the placers

- **Deterministic placer:**
  - seeds, descents, multistart and layer-guided moves work unchanged;
  - the star closed form ranks sites by round-trip move time instead of
    Euclidean length, so it stays exact under F_time;
  - the exhaustive isometry-class route was removed with F: F_time's
    nearest-neighbour tour breaks ties by stop order, so it is not exactly
    invariant under reflections, and the route's optimality proof does not
    hold.
- **FastSA normalises the cost it anneals** (since 2026-10-20). The cost
  is divided by its mean magnitude over the start layout and stage 1's random
  layouts, which costs no extra evaluations. A random layout then scores about
  1, as Fast-SA prescribes (Chen & Chang 2006). This is needed because the
  schedule is not scale-free: T₁ = Δ_avg/|ln P| is proportional to the cost,
  and T = T₁⟨Δcost⟩/(n·c) to its square. Multiplying the cost by s has the
  same effect as dividing |ln P| by s, so the cost's units act as a hidden
  temperature knob. With normalisation, P (`initial_accept_prob`), c and
  `stage3_temperature_threshold` mean the same thing on every circuit.
  `FastSAResult` costs are normalised; `cost_scale` converts back, and the
  `--log` CSV writes `best_cost` in µs.
- **FastSA defaults** (tuned in §5): P = 0.999, c = 100 and
  `stage3_temperature_threshold` = 0, so stage 3 runs all
  `stage3_max_iterations`. A config's `Fastsa_fill` block still overrides
  them.
- **Speed.** When `alignment_conc` is −1 and the AOD span covers the whole
  array (`max_dimension` ≥ `dimensions`), every gate fits the first load and
  the mover choice cannot change the stops. That covers every qLDPC config and
  most others. Each layer is then priced directly as one load whose stops are
  its distinct displacements, with no grouping.

  Otherwise the general grouping runs. It is still exact, and keeps a load's
  bounding box incrementally. It also computes the router's alignment score
  once instead of once per mover choice, because swapping the mover negates
  the score exactly.

  `tests/test_placement_cost.py` checks both paths against the original
  grouping for exact equality.

  | Config | Before | After |
  |---|---|---|
  | bb144 (`bb144_n_fastsa_k1000.json`) | 33.1 ms | 3.7 ms (×9.0; the original F took 16.5 ms) |
  | QFT-29 (`Scheduler_Test/INPUT/qft_n29.json`) | 4.5 ms | 0.61 ms (×7.4) |
  | QFT-18 (`Scheduler_Test/INPUT/qft_n18.json`, `alignment_conc` 0, general path) | 1.5 ms | 1.0 ms (×1.5) |

  An incremental update (re-score only the layers or sections touching the
  moved qubits) would be the next speedup.

## 3. QFT-29 comparison (F vs F_time, 2026-10-20)

Run on 2026-10-20 with `inputs/algorithms/qft_n29_init.json`. That's
`reorder_cz` off, so F_time is in fixed-layer mode; trapezoidal moves at
0.55 m/s and 2750 m/s²; `alignment_conc: -1`. The scheduler is the same for
every run (`tour_ordering: "tsp"`). FastSA uses `main`'s default
`Fastsa_fill` (about 1,300 evaluations, seed 0); the deterministic placer
uses its default budget. Every final layout is scored under both costs.

| Placement | Cost optimised | Texec | F | F_time | Wall time |
|---|---|---|---|---|---|
| given layout (`qft_n29_coords.json`) | – | 139702.7 µs | 3.5977 | 173663.3 µs | 2.1 s |
| deterministic | `fastsa` | **97262.0 µs** (−30.4%) | 2.8237 | 139787.1 µs | 2.8 s |
| deterministic | `time` | **97262.0 µs** (−30.4%) | 2.8237 | 139787.1 µs | 93.6 s |
| FastSA | `fastsa` | 151032.6 µs (+8.1%) | 4.2830 | 188918.8 µs | 2.8 s |
| FastSA | `time` | 154895.4 µs (+10.9%) | 4.5551 | 190011.8 µs | 9.2 s |

- **The deterministic placer reaches the same layout under both costs.** It
  starts from different best seeds (`rcm | w=5 snake` under F,
  `rcm | w=6 snake transposed` under F_time) and ends at the same serpentine
  of QFT's qubit chain, a local optimum of both. Under F_time it used 20,506
  evaluations and 5 descents, against 11,798 and 3.
- **QFT-29 does not separate the two costs.** F and F_time both rank the four
  distinct layouts exactly as Texec does. The placer matters more than the
  cost here. With `main`'s default settings, FastSA ends worse than the given
  layout under either cost.
- **F_time runs 23–44% above Texec** on these layouts (the ratio is 1.23–1.44),
  as expected from the terms it leaves out.
- **Cost of F_time:** 33× the wall time of the deterministic F run. An
  incremental update is needed before F_time is practical on larger circuits.

Where the two costs should differ is circuits whose schedule depends on
signed displacements, `reorder_cz` re-layering or alignment-based grouping.
Examples are the qLDPC configs (ancilla movers, `reorder_cz: true`),
especially bb_n90 and bb_n360, where lower F gave *slower* schedules than
row-major in the collaborator's runs, and any config with
`alignment_conc` > −1.

## 4. Benchmark: is F_time a better cost? (2026-10-20)

Script: [`benchmarks/cost_fidelity.py`](benchmarks/cost_fidelity.py). Per-layout data:
`benchmarks/results/cost_fidelity_2026-10-20.csv`.

- **Circuits:** 9 general circuits, no structured codes. QFT-18 and QFT-29,
  BV-50 and VQE-30 come from the repo. QAOA-24 (p = 2, 3-regular), random-24
  (150 CX), quantum volume 20 (depth 6), Heisenberg-24 (3 Trotter steps) and
  an 18-qubit ripple-carry adder are generated into `benchmarks/circuits/`.
- **Physics:** two settings. "Slow" is QFT-29's (0.55 m/s, trapezoidal,
  `T_reuse` 3). "Fast" is bb144's (10 m/s, cubic spline, `T_reuse` Inf).
- **Fixed settings:** `alignment_conc` −1 and `tour_ordering: "tsp"`.
- **Job size:** about 145 scheduled layouts per job. Each job takes 1–7
  minutes on one core.

### Fidelity: how well each cost orders layouts like the scheduler

| Metric (median over 18 jobs) | F | F_time |
|---|---|---|
| Spearman ρ over the sample | 0.898 | **0.922** |
| Local sign agreement (one-move neighbours) | 0.822 | **0.870** |
| Selection regret (best by cost vs truly best) | 4.0% | **0.7%** |

- **F_time is the better predictor overall, but not everywhere.** Its local
  agreement is lower on adder (fast), BV-50 (fast), Heisenberg (fast) and
  QV-20 (fast).
- **Both costs are weak on BV-50.** That's a star circuit with many idle
  qubits: ρ ≈ 0.55–0.63 and local agreement 0.40–0.73.

### Outcome: scheduled time after optimising each cost

**Deterministic placer** (budget 20,000; it only compares costs, so this is the clean
test). F_time gave the faster schedule in 9 jobs, F in 5, with 4 ties. The
median ratio is 0.998, and the spread is large in both directions:

| Biggest gains with F_time | ratio | Biggest losses with F_time | ratio |
|---|---|---|---|
| Heisenberg-24, fast | 0.684 | adder-18, fast | 1.170 |
| Heisenberg-24, slow | 0.776 | QV-20, fast | 1.052 |
| QAOA-24, fast | 0.802 | BV-50, fast | 1.036 |
| QFT-18, slow | 0.880 | QFT-29, fast | 1.023 |

Running the deterministic placer under both costs and keeping the result
with the lower F_time picks the faster schedule in 14 of 18 jobs. That gives
the same median, 0.998 of F alone, with the same extremes (0.684–1.170).

**FastSA** (3 seeds, `main`'s default settings, median shown):
- annealed with both costs relative to the start, F_time wins 8 jobs and F
  wins 9;
- F *as shipped*, on its own scale, beats F_time in 16 of 18 jobs.

FastSA's temperature scales with the square of the cost (§2), so its results
depend on how the cost happens to be scaled. Even F annealed relative to the
start is clearly worse than F as shipped. FastSA comparisons therefore
measure its tuning more than the cost.

### Ease

| | F | F_time |
|---|---|---|
| Per evaluation (median) | 0.08 ms | 1.2 ms (up to 7.7 ms on QFT-29) |
| Deterministic run, QFT-29 | 1.5 s | 96–126 s |

### Verdict so far

Outcome ratios (F_time result / F result, scheduled time):

| | Geometric mean | Median | Better / worse | Range |
|---|---|---|---|---|
| Deterministic placer | **0.948** | 0.998 | 9 / 5 (4 ties) | 0.684–1.170 |
| FastSA, both costs relative to start | 1.002 | 1.002 | 8 / 9 | 0.913–1.088 |
| FastSA, F_time (relative) vs F as shipped | 1.184 | 1.175 | 2 / 16 | 0.933–1.572 |

- **The deterministic placer, which only compares costs, gets about 5% faster
  schedules on average with F_time.** The median is close to 1 because about
  half the jobs barely change; the gains are large where they occur
  (Heisenberg, QAOA, QFT-18). It is not consistent: the adder loses 17%.
- **FastSA does not benefit, but that's about FastSA, not the cost.** With
  both costs on the same scale they tie; F_time only loses against F as
  shipped, because FastSA's temperature schedule is effectively tuned to F's
  scale (§2).
- **F_time costs 15–60× more per evaluation.**
- **Its losses cluster in the fast physics,** which uses `T_reuse: Inf`. One
  hypothesis, not yet verified, is that the scheduler's load reuse across
  layers, which F_time does not model, dominates there.

## 5. FastSA tuning under normalisation (2026-10-20)

Script: [`benchmarks/fastsa_tuning.py`](benchmarks/fastsa_tuning.py). Data:
`benchmarks/results/fastsa_tuning_2026-10-20.csv`.

- **Grid:** P ∈ {0.99, 0.999, 0.9999, 0.99999} × stage-3 threshold
  ∈ {1e-3, 0}, where 0 means stage 3 runs all its `stage3_max_iterations`.
  c stays at 100, since it affects only stage 2, which is greedy by design.
- **Runs:** both costs, 3 seeds each, on the same 18 jobs as §4, everything
  else at `main`'s defaults.
- **Score:** geometric mean of the median scheduled time over seeds, relative
  to the *old shipped* FastSA (F unnormalised, P 0.99, threshold 1e-3; from
  the §4 data).
- **Selection:** settings were chosen on 5 training circuits (QFT-18,
  QAOA-24, random-24, adder-18, VQE-30, both physics settings) and reported
  on 4 held-out circuits (QFT-29, BV-50, QV-20, Heisenberg-24, both physics
  settings).

| cost | P | threshold | train | held-out | mean stage-3 iterations |
|---|---|---|---|---|---|
| F | 0.99 | 1e-3 | 1.268 | 1.114 | 9 |
| F | **0.99** | **0** | **1.026** | **0.996** | 1000 |
| F | 0.999 | 0 | 1.036 | 1.015 | 1000 |
| F | 0.9999 | 0 | 1.175 | 1.077 | 1000 |
| F_time | 0.99 | 1e-3 | 1.243 | 1.111 | 2 |
| F_time | 0.99 | 0 | 1.043 | 0.963 | 1000 |
| F_time | **0.999** | **0** | **1.040** | **0.948** | 1000 |
| F_time | 0.9999 | 0 | 1.087 | 1.005 | 1000 |

Settings with P ≥ 0.9999 behave the same at either threshold, and 0.99999
is worse than 0.9999.

- **Under normalisation, threshold 1e-3 effectively switches stage 3 off**
  (2–9 iterations), which costs 11–27%. Stage 3 has to run its full
  iteration cap: threshold 0.
- **Normalised, tuned F performs like the old shipped F** (training 1.026,
  held-out 0.996). Normalisation makes the settings portable but does not by
  itself improve F.
- **Tuned F_time against tuned F:** geometric mean 0.952 on held-out circuits
  (7 better, 1 worse) and 0.986 over all 18 jobs (13 better, 5 worse). Each
  setting has 3 seeds, so differences of a few percent are partly noise.
