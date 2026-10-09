Implementation references:

- [`naive_dag` dynamics](../src/naive_dag/dynamics.py)
- [`naive_n_dag` dynamics](../src/naive_n_dag/dynamics.py)
- [`naive_dag` DAG preparation](../src/naive_dag/dag_helper.py)
- [`naive_n_dag` DAG preparation](../src/naive_n_dag/dag_helper.py)

## 1. Coordinate model

The schedulers use integer grid coordinates `(row, column)`.

- An **SLM trap** is an even-even coordinate: both row and column are even.
- Every other coordinate is an **AOD position**.
- Horizontal AOD highway movement is legal only on an odd row.
- Vertical AOD highway movement is legal only on an odd column.
- An odd-odd coordinate is a highway intersection where movement may change axis.
- For configured SLM dimensions `[R, C]`, initial and unloading trap coordinates lie within rows `0..2R-2` and columns `0..2C-2`.
- AOD highway coordinates may extend outside that rectangle. The current model does not impose a finite physical AOD travel boundary.

A straight movement segment is judged by its complete interior, not only its endpoints. A segment may not pass through an even-even SLM coordinate. For example, `(6,1) -> (6,5)` is illegal because horizontal movement is on even row 6 and crosses SLM traps.

## 2. Circuit preservation

The following rules apply to both schedulers.

- Input SWAP operations are removed while constructing the scheduling DAG.
- SWAPs must not appear in scheduling layers or emitted schedules.
- Neither scheduler may insert a routing SWAP or replace an omitted SWAP with a decomposition.
- If a SWAP unexpectedly reaches a layer, it is omitted and the DAG filtering path should be investigated.
- Circuit comparison is performed against the supported input operation sequence **after SWAP removal**.
- Every remaining supported one- and two-qubit operation must appear exactly once. No operation may be added, duplicated, or omitted.
- Gate identity, parameters, measurement destinations, and per-qubit operation order must be preserved.
- Optional exception for `naive_n_dag` QASM input: `reorder_cz=true` or
  `reorder_cz_by_displacement=true` permits
  exchanging unconditional CZs within commuting blocks before scheduling.
  Preprocessing preserves every operation occurrence and checks that each
  quantum/classical wire changes only by permutations within consecutive CZ
  runs. It does not cross non-CZ wire boundaries. The movement scheduler then
  preserves exact per-qubit order against the prepared, SWAP-filtered circuit.
  With both passes disabled, original input ordering is retained. The
  displacement option defaults to the value of `reorder_cz`. Both options
  reject conditional gates/control flow and `step_order` input.
- Operations on disjoint qubits may be reordered or emitted in the same timestep when the scheduler's batching rules permit it.
- A gate may not use the same qubit twice.
- Simultaneous gates must have disjoint operands.
- Measurements retain the classical register and destination, for example
  `measure q[2] -> c_x[5];`. Equal local indices in different registers remain distinct.
- Barriers are scheduling metadata and are not emitted as physical operations.
  Every barrier is a hard global epoch boundary: all operations before it must
  be emitted before every operation after it, even when their operands are
  disjoint. Reduced DAGs retain explicit empty separator layers for this rule.
- The supported physical scheduling model handles one- and two-qubit operations. Unsupported higher-arity operations must not be silently scheduled as if they were supported.

## 3. Initialization and state tracking

- Every logical qubit is initialized exactly once.
- Initial coordinates are unique, in-bounds even-even SLM traps.
- The initialization line records the actual placement before scheduling starts.
- Movement planning must not mutate or replace the initialization metadata written at `T=0`.
- Each scheduler tracks the current coordinate of every atom throughout the complete schedule.
- Every movement action starts at the coordinate where that atom's preceding action ended.
- Candidate schedules use independent position state. Changes made while evaluating one candidate must not leak into another candidate or into the original placement objects.

## 4. Loads, unloads, and ordinary movement

- A `load` starts at an even-even SLM trap and ends at a non-even-even AOD position.
- An `unload` starts at an AOD position and ends at an even-even SLM trap.
- Every load and unload is exactly one cardinal grid step.
- Direct SLM-to-SLM movement is forbidden.
- An ordinary `move` has AOD positions at both endpoints.
- Ordinary movement is axis-aligned and follows the parity highways described in Section 1.
- Zero-length movement events are omitted and cost no time.
- A movement endpoint may not coincide with another atom.
- A highway route may not pass through an occupied SLM trap; the stronger parity rule normally excludes every SLM trap from an ordinary segment regardless of occupancy.
- An unload destination must be vacant at the moment of unloading. It is allowed to have been occupied earlier in the schedule.
- Every schedule ends with no atom left loaded in the AOD.

The current discrete model checks grid-position collisions and legal highway interiors. It does not model finite trap extent, analog collision envelopes between moving paths, laser crosstalk, or finite AOD travel boundaries.

## 5. Two-qubit interaction geometry

At every emitted two-qubit gate:

- Exactly one operand is in the AOD.
- Exactly one operand remains in an even-even SLM trap.
- The two operands are exactly one cardinal grid step apart, so their Manhattan distance is `1`.
- The loaded operand must be the atom tracked by the scheduler as active in the AOD.
- A gate with both operands loaded is forbidden.
- A gate with both operands in SLM traps is forbidden under this movement model.

Single-qubit operations and measurements preserve circuit order. They do not, by themselves, require an atom that is needed by the immediately following two-qubit gate to be unloaded.

## 6. Common timing rules

For `movement_shape: "trapezoidal"`, the physical distance of a grid
displacement `(dr, dc)` is

```text
distance = hypot(dr, dc) * rydberg_radius / 2
```

Each emitted movement segment starts and stops at rest. With maximum velocity `v` and acceleration `a`, its duration is

```text
ramp_distance = v^2 / a

if distance > ramp_distance:
    time = 2v/a + (distance - ramp_distance)/v
else:
    time = 2 * sqrt(distance/a)
```

For `movement_shape: "Cubic_Spline"`, movement uses the minimum-jerk timing
model from *Matching_Generalized*. With physical axis distances
`delta_x = abs(dr) * rydberg_radius / 2` and
`delta_y = abs(dc) * rydberg_radius / 2`, its duration is

```text
time = sqrt(6 * delta_x / a) + sqrt(6 * delta_y / a)
```

The cubic-spline profile does not use `max_velocity` in its movement-time
calculation. Both supported profiles retain the same transfer, segmentation,
batching, and gate-time rules below.

Additional timing rules:

- A transfer charge is added once per actual load or unload batch.
- Ordinary highway movement has no transfer surcharge.
- Shared rigid movement in `naive_n_dag` is charged once per simultaneous batch, not once per transported atom.
- When multiple atoms move in one valid rigid batch, batch duration is the longest movement duration in that batch. Valid batches share the same displacement, so these durations should agree.
- Consecutive same-direction highway pieces that serialize as one segment are timed as one segment.
- A turn, gate, load, unload, direction reversal, or otherwise separate emitted segment creates a stop/start boundary.
- A one-qubit pulse identity is the gate name plus its parameters. An identical pulse applied to disjoint atoms in one timestep is charged once.
- Distinct pulse identities in the same timestep are charged separately.
- Each one-qubit pulse batch costs `average_single_gate_time + t_switch`.
- Each two-qubit pulse batch costs `average_two_gate_time + t_switch`.
- All measurements in one timestep form one measurement batch costing
  `measurement_time + t_switch`; `measurement_time` defaults to `0 milliseconds`.
- Reported complete-schedule time includes all emitted movement, transfers, gates, and final unloading/return required by that scheduler.

Candidate schedules may be compared only when timed under the same model and from the same initial placement.

## 7. Rules specific to `naive_dag`

### 7.1 One moving atom

- At most one atom is active in the AOD at any time.
- A new atom may not load until the active atom has unloaded.
- Only the active atom may perform an ordinary movement or unload.
- The scheduler chooses one operand of each two-qubit gate as the mover.

### 7.2 Mandatory immediate reuse

- If the next two-qubit gate uses the currently loaded atom, that atom remains in the AOD.
- Exception: with `prefer_moving_ancillas=true`, a loaded data atom returns
  before a data-ancilla gate so that the reset-identified ancilla can move.
- It must not be unloaded and immediately loaded again for that next gate.
- Intervening one-qubit operations do not cancel this rule.
- The atom may move through legal parity routes from its current interaction site to the next partner.
- It unloads only when the next two-qubit gate requires a different mover or when the circuit is complete.

### 7.3 Changing homes is allowed

`naive_dag` may change an atom's SLM home during execution.

- When unloading is necessary, the atom may return to its previous home or choose another currently vacant SLM trap.
- The destination need only be vacant at unloading time; prior occupancy does not disqualify it.
- The new trap becomes that atom's current home for later planning.
- Final positions may differ from initialization.
- Relocation is chosen for speed, not merely because a trap appears first in grid order.

The relocation heuristic compares the previous home with a bounded set of vacant destinations. By default it scores at most eight destinations and looks ahead through at most eight upcoming two-qubit gates or two upcoming uses of the atom being unloaded, whichever limit is reached first. The score includes the legal route and unload plus simulated future movement and transfers. Ties favor the previous home.

The optional configuration keys are:

| Key | Default | Meaning |
|---|---:|---|
| `relocation_enabled` | `true` | Enable the relocation candidate |
| `relocation_candidates` | `8` | Maximum unload destinations scored |
| `relocation_lookahead_gates` | `8` | Maximum future two-qubit gates simulated |
| `relocation_lookahead_uses` | `2` | Maximum future uses of the atom simulated |

### 7.4 Complete-plan protection

Standalone `naive_dag` builds complete plans from the same initial placement:

- `previous_home`: mandatory immediate reuse, followed by unloading to the atom's previous home whenever unloading is required.
- `relocation`: mandatory immediate reuse, followed by lookahead-based destination selection whenever unloading is required.

Both plans are validated and timed. The lower-duration complete plan is emitted; ties favor `previous_home`. Therefore relocation cannot make the selected complete schedule slower than the corrected previous-home baseline under the common timing model.

## 8. Rules specific to `naive_n_dag`

### 8.1 Immutable homes and final restoration

For native `naive_n_dag` candidates, the initial SLM positions are immutable homes.

- Native candidates may move atoms during execution but do not permanently change their homes.
- Every active AOD load is completely returned and unloaded when a reset is required.
- At circuit completion, every atom is back at its own original home.
- Final positions equal initialization exactly.

This differs intentionally from standalone `naive_dag`, which may finish at a changed valid placement.

### 8.2 Rigid AOD-load behavior

`naive_n_dag` may move one atom or a group of atoms in the AOD.

- A new load may begin only when there is no existing active load.
- Every atom in an active load moves together with the same displacement.
- The entire active load must move or unload together. A subset may not move or unload independently.
- Loaded atoms retain a common displacement relative to their homes so that a rigid return is possible.
- A load must fit within the configured `max_dimension` AOD span.
- Atom coordinates remain unique after every movement batch.
- `parallel=false` restricts scheduling to one mover per group.
- `parallel=true` permits grouping gates when the AOD span and vector-alignment rules allow it.
- The signed alignment threshold is controlled by `alignment_conc`.
- Within a selected group, at most 24 unique signed displacement vectors
  `partner_position - mover_home` are ordered by an exact Held-Karp path using
  the configured movement-time model. The path starts at the shared current
  displacement `mover_position - mover_home` (zero for a fresh load) and ends
  at home `(0, 0)`, including for reused loads. Edge costs use the differences
  between these home-relative targets. This is an approximate routing objective:
  interaction-side choices and parity detours are handled during shuttling,
  not added to the search space. All gates with an
  identical displacement remain consecutive and retain their original relative
  order. More than 24 unique displacements emits a warning and falls back to
  the greedy alignment ordering on current-relative vectors.
- Ordinary group routing then applies a fixed-order four-side dynamic program
  using the existing parity router, for every supported 2Q gate type regardless
  of CZ preprocessing flags. This also applies to one-mover groups and after
  the above-24-target alignment fallback. Fresh groups optimize their load
  direction; reused groups start at the actual rigid AOD offset without another
  load. The cost includes the eventual return home, but execution leaves the
  load active until the ordinary planner decides to return it. Gates are moved
  only within disjoint groups, never across noncommuting dependencies.

### 8.3 Layer and reuse rules

- A two-qubit DAG layer contains gates with disjoint operands.
- With CZ preprocessing enabled, bipartite commuting blocks use optimal
  edge coloring (Delta colors, via regularization and perfect matchings);
  non-bipartite blocks use greedy coloring. Rebuilding the DAG in color order
  permits the ordinary layer extractor to compact independent gates across
  colors. It does not guarantee minimum movement time or fixed color boundaries.
- Before placement, compare original and proposed extracted two-qubit layer
  counts. If the proposed count is larger, retain the original full circuit,
  single-qubit context and validation reference. Accept equal or smaller
  counts. This depth safeguard does not guarantee shorter physical execution.
- After placement, `reorder_cz_by_displacement` optionally repacks reachable
  commuting CZ blocks using signed home-to-partner displacement, with the same
  reset-identified ancilla preference as dynamics (otherwise the first operand
  defines the proxy mover). Equal-displacement disjoint matchings are kept
  intact and packed largest-first into compatible layers. Conflicting gates,
  including repeated occurrences, remain separate. Near-home layers run first
  within each commuting block. This is a heuristic, not an optimal pulse solver.
- The placement pass preserves operation occurrences, quantum/classical wire
  boundaries and global barrier epochs. Non-CZ operations are not commuted on
  shared wires. It rebuilds the single-qubit context and validation reference
  together with the explicit layers; no later ASAP extraction may split the
  displacement bundles. AOD span/alignment and interaction legality are still
  checked by grouping, shuttling and replay.
- Packed layerings may increase logical depth. Complete original and packed
  schedules are compared by actual duration, retaining the original on ties.
- The optional `phase_tour` candidate bypasses the packed layer boundaries:
  it orders the unique home-relative displacements across a whole reachable
  commuting CZ block. A consistent bipartition, honoring preferred movers,
  keeps exactly one operand of every interaction in a single rigid load.
  All movers, including atoms first used later in the tour, load together.
  Partners remain stationary; the complete load returns home after the block.
- Phase tours use the same at-most-24-target displacement Held-Karp objective
  (warn and use alignment above 24), without adding interaction-side states
  to the subset search. After ordering, a separate fixed-order DP retains four
  endpoint states per visit and evaluates 16 transitions between visits, plus
  four load/unload directions at the endpoints. It minimizes parity-routed
  motion for that order, not jointly over all visit orders or all pulse costs.
  The existing parity router, highway rules and replay validator are unchanged.
  Planned routes arrive adjacent to every gate before invoking its existing
  executor, so shuttling need not choose another side. Forward and reverse
  traversals use only the configured side policy (DP by default, greedy only
  when requested) and are physically replayed; the lowest complete duration
  is selected. Every equal-target visit is split
  into disjoint pulses when necessary to preserve repeated gate occurrences.
- Non-bipartite blocks, contradictory mover preferences, oversized full loads,
  or incompatibility with `parallel`/`T_reuse` reject only the phase candidate.
  Non-CZ operations, reset, measurement and global barriers remain present.
  The reported phase layer count is the number of disjoint visit groups, not
  the count of previous packed layers. The selected complete duration, not
  that count, determines whether the candidate is accepted.
- Grouping decisions use the candidate's actual execution state after any proposed reset.
- Reuse may execute only gates having exactly one operand in the existing active load.
- With `prefer_moving_ancillas=true`, a data-ancilla gate may reuse its loaded
  operand only when that operand is the reset-identified ancilla. New groups
  likewise choose the ancilla as mover.
- A gate whose two operands are both loaded cannot execute until the load has returned and an appropriate load is created.
- Reuse and reset alternatives are built on independent state copies.
- The local reuse decision compares each option with a hypothetical closing return so both costs have the same all-home endpoint.
- Reuse wins only when its closed cost is strictly lower; a tie favors reset.

`T_reuse` controls eligibility for retaining extra loaded atoms:

- The horizon is counted in two-qubit DAG layers, not microseconds or emitted timesteps.
- For `phase_tour`, the horizon is counted in its disjoint displacement visits;
  extra atoms in the full load must satisfy the same retention predicate at
  every visit. `Inf` permits retaining the entire load until the block ends.
- A finite positive value permits an extra loaded atom only when its next use is known within the horizon.
- `Inf` permits indefinite eligibility but does not override legality or cost comparison.
- `T_reuse=0` permits identical-load continuation but forbids retaining extra loaded atoms.
- A larger horizon does not guarantee a monotonically shorter result because the reuse planner is greedy.

### 8.4 Configuration-controlled scheduling

`naive_n_dag.schedule_circuit` builds one layered schedule using the supplied
configuration, rather than comparing complete scheduling policies:

- `T_reuse` controls legal, cost-tested load retention, including identical-load
  continuation at zero. Returning home when necessary remains part of execution.
- Held-Karp orders unique displacements up to `_MAX_EXACT_DISPLACEMENT_TOUR`
  (24). Only exceeding that limit triggers a warning and alignment ordering;
  there is no group-ordering configuration.
- Side DP is the default. Only `_side_ordering="greedy"` selects greedy sides.
- The result is named `reuse` or `greedy_reuse`, respectively.
  `candidate_times` contains only that configured run.
- Replay validation and common-model timing remain mandatory. Planning or
  validation errors propagate; the scheduler does not try another policy.

Local reset-versus-reuse cost comparisons remain within the configured policy.
No complete no-reuse, alignment, sequential-home, or legacy baseline is built.
Held-Karp is exact for its reduced displacement objective, not the complete
physical duration; there is no duration guarantee against uncomputed policies.

With placement-aware preprocessing enabled, `main.py` still compares original
layers, packed layers (with the `displacement_` name prefix), and the optional
`phase_tour`. All use the same configured side and retention policies.
Each is replayed against its checked reference order. Phase tours compare both
traversal directions without trying another side policy. Phase-wide side plans
bypass group-level reoptimization when emitting already-positioned gates.

Gate occurrences and legal CZ permutations are checked before full replay.
Invalid phase candidates are reported and excluded; ties retain the previously
selected schedule. This preprocessing comparison is independent of the removed
scheduling-policy sweep.

### 8.5 Legacy adapter

`_legacy_baseline` remains an internal adapter for standalone `naive_dag`
routing on a copied placement, adding legal restoration to the original homes.
It is not called by `schedule_circuit` and provides no bound on its selected
duration.

## 9. Serialization rules

- `T=0` contains the true initialization.
- Subsequent printed timestep numbers are consecutive.
- Every nonempty printed timestep contains either a gate batch or one movement batch.
- Movement labels are determined by endpoint parity: SLM-to-AOD is `load`, AOD-to-SLM is `unload`, and AOD-to-AOD is `move`.
- A qubit appears at most once in one movement batch.
- `naive_dag` emits one moving atom per movement timestep.
- `naive_n_dag` batches ordinary movements only when they share the required rigid displacement and action kind. Load and unload boundaries remain explicit.
- Serialized gate text contains all parameters and measurement destinations needed for independent replay.
- The serialized `final_time` must equal the duration recomputed from the serialized actions under the applicable common timing model.

## 10. Validation requirements

Before a complete schedule is accepted or written, it must be replayed from its true initialization and checked for:

- Initialization uniqueness, parity, and bounds.
- Movement continuity and correct action labels.
- Legal cardinal transfers and legal highway interiors.
- Active-load ownership and solver-specific one-atom or rigid-group behavior.
- Vacant movement endpoints and unload destinations.
- Correct two-qubit interaction geometry.
- Circuit identities, parameters, multiplicities, measurement destinations, and per-qubit order against the SWAP-filtered reference.
- No emitted SWAPs.
- No atom remaining active in the AOD at completion.
- The solver-specific terminal rule: changed valid homes are allowed for standalone `naive_dag`; exact original-home restoration is required for `naive_n_dag` and its accepted adapted candidates.
- Agreement between internal events, serialized actions, and reported duration.

## 11. Scope limitations

These rules define the repository's current discrete scheduling model. They do not claim to model arbitrary QASM control flow, finite optical-trap extent, crosstalk, analog trajectory intersections away from represented grid points, hardware-specific measurement duration, or finite AOD travel bounds. Such constraints require separate rules and validation before being treated as hardware guarantees.
