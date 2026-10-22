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
  A barrier is a hard boundary for the qubits it names: every operation on
  one of them before the barrier must be emitted in an earlier timestep than
  every operation on one of them after it. Operations on other qubits may
  cross it, so a barrier on every qubit is a global boundary. Reduced DAGs
  keep the barrier on its own qubits; it takes the layer after their latest
  two-qubit gate (as `DAGCircuit.layers()` places it), which may also hold
  gates on other qubits. Barriers are never widened.
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

### 5.1 Rydberg-radius pulse separation (`naive_n_dag`)

`lattice_spacing` (config) is the SLM trap pitch; one grid step is
`lattice_spacing / 2`. The Rydberg radius is the fixed constant
`RYDBERG_RADIUS = 15 micrometers` in `naive_n_dag/dynamics.py`.

- A gate pair is one grid step apart, so `lattice_spacing / 2` must be strictly
  smaller than the Rydberg radius.
- Two-qubit gates may share one timestep (one Rydberg pulse) only when every
  atom of each gate is at least `RYDBERG_RADIUS` from every atom of every other
  gate in that timestep. Atoms exactly at the radius do not conflict. Atoms not
  participating in the pulse are not considered.
- Gates that would conflict are emitted in separate consecutive timesteps
  while the AOD load stays in place. A gate is placed in the earliest such
  timestep in which it conflicts with nothing and shares no qubit with that
  timestep or any later one. If none fits, the current stop's timesteps and
  the new gate are recolored together (DSATUR graph coloring of the conflict
  graph, polynomial time) and the result is kept when it uses fewer timesteps
  than opening a new one. DSATUR is exact whenever two pulses suffice; for
  denser conflict graphs (for example, small `lattice_spacing`) it is a
  heuristic, since the exact minimum is NP-hard. Gates are never packed across
  a QASM barrier or retained-load routing stage.
- Each such timestep is a separate two-qubit pulse batch and is charged
  `average_two_gate_time + t_switch` (Section 6). Shuttling routes, side
  choices, and visit order are unchanged by this rule.

Single-qubit operations and measurements preserve circuit order. They do not, by themselves, require an atom that is needed by the immediately following two-qubit gate to be unloaded.

## 6. Common timing rules

For `movement_shape: "trapezoidal"`, the physical distance of a grid
displacement `(dr, dc)` is

```text
distance = hypot(dr, dc) * lattice_spacing / 2
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
`delta_x = abs(dr) * lattice_spacing / 2` and
`delta_y = abs(dc) * lattice_spacing / 2`, its duration is

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

- A new load may begin only when there is no existing active load, except
  for held atoms (below).
- Held atoms: when an unload step is followed by a load step and an atom is
  in both, it stays in the AOD. It rides the unload step onto its own home
  site and the load step off again, without a transfer, and is written as
  `move q[i] (r,c) : (r',c')` inside that step's line. Only atoms that
  actually change traps are written as `load`/`unload`. An atom measured or
  reset between the two steps is released and reloaded as before. A step in
  which every atom is held charges motion but no `transfer_SLM_AOD`.
- Every atom in an active load moves together with the same displacement.
- The entire active load must move together. A subset may not move independently; in an
  unload step every loaded atom either unloads or is held (below).
- Loaded atoms retain a common displacement relative to their homes so that a rigid return is possible.
- A load must fit within the configured `max_dimension` AOD span.
- Atom coordinates remain unique after every movement batch.
- Gates are always grouped into one AOD load when the AOD span (`max_dimension`)
  and vector-alignment rules allow it. `max_dimension: [1, 1]` has zero span,
  so it restricts scheduling to one mover per load. (The former `parallel` key
  was removed; a config that sets it is rejected with an error.)
- The signed alignment threshold is controlled by `alignment_conc`.
- Within a selected group, the unique signed displacement vectors
  `partner_position - mover_home` are ordered by a TSP tour using the
  configured movement-time model. The path starts at the shared current
  displacement `mover_position - mover_home` (zero for a fresh load) and ends
  at home `(0, 0)`, including for reused loads. Edge costs are the cheapest
  parity-routed motion between any interaction sides of the two targets
  (Section 8.2, layered graph). All gates with an identical displacement remain
  consecutive and retain their original relative order. There is no size limit.
- The tour is solved by Concorde (`pyconcorde`). If pyconcorde is missing, a
  warning is issued and `fast_tsp` is used. Graphs with 4 or fewer nodes, which
  Concorde cannot solve, always use `fast_tsp`, bounded by `tsp_time_limit`
  seconds per solve (default `0.05`). If Concorde fails (an error, or a tour
  failing validation), `fast_tsp` is tried next, with a warning. Only when no
  solver yields a valid tour is the group ordered by the alignment heuristic
  `_sort_group_by_alignment` instead, with a warning. The
  two methods are never both run and compared.
- Ordinary group routing then applies a fixed-order four-side dynamic program
  using the existing parity router, for every supported 2Q gate type regardless
  of CZ preprocessing flags. This also applies to one-mover groups and after
  an alignment fallback. Fresh groups optimize their load
  direction; reused groups start at the actual rigid AOD offset without another
  load. The cost includes the eventual return home, but execution leaves the
  load active until the ordinary planner decides to return it. Gates are moved
  only within disjoint groups, never across noncommuting dependencies.

- After the ordinary planner chooses load/reuse boundaries, continuous loads
  spanning multiple DAG layers are jointly rerouted in `dynamics.py`.
  Each graph city is a `(layer, displacement)` pair. A load terminal (the
  current rigid offset) and an unload terminal (home) act as layers before the
  first and after the last. Same-layer and adjacent-layer edges have symmetric
  costs; nonconsecutive edges are absent (infinite cost, never traversed). The
  unload-to-load edge is free and closes the cycle, so the matrix is a
  symmetric TSP instance usable by external solvers such as Concorde.
- Every layer-crossing edge, including load and unload, adds a penalty greater
  than the movement cost of a known feasible tour. A Hamiltonian cycle crosses
  each cut an even number of times, so a tour that splits or revisits a layer,
  or avoids the closing edge, pays at least two extra penalties and cannot be
  optimal. Every solver's tour is validated:
  Hamiltonian, existing edges only, closing edge used, layers in order. The
  penalty is an optimization device, never emitted schedule time.
- Edges are priced at the cheapest parity-routed pair of interaction sides
  (load/unload with their best transfer direction). The matrix is symmetric
  with one node per stop; the exact fixed-order side DP then fixes sides for
  the chosen order. Weights are scaled to integers for the solvers, with a
  large finite weight for absent edges.
- With `reorder_cz` or `reorder_cz_by_displacement` enabled, consecutive
  CZ-only layers in one retained load with no 1Q context, barrier or
  barrier-emptied layer between them form one graph stage: unconditional CZs commute, so
  their stops may be visited in any order. Replay validation then compares
  per-wire traces up to permutations of maximal unconditional CZ runs, as in
  CZ preprocessing; otherwise exact per-wire order is still required.
- Empty layers are skipped when numbering graph stages. Intervening 1Q contexts
  fence stages and remain in their original logical order. Repeated targets in
  different stages are separate cities. Gates with identical targets within a
  stage retain occurrence order and remain consecutive.
- If the tour fails, warn and apply alignment ordering separately within each
  stage, never mixing stages. Validate the returned permutation and stage order
  before emitting movements.
- The side DP applies across the retained-load interval: it prices one load
  and one final return, not a return between layers. No load membership or retention decision
  changes during rerouting. Physical replay still checks the complete circuit.
  Exactness concerns the layered displacement proxy, not the joint physical
  order/side/load-selection problem.

### 8.3 Layer and reuse rules

- A two-qubit DAG layer contains gates with disjoint operands.
- The CLI completes initial placement before CZ displacement reordering.
  Within each reachable commuting block, gates with identical signed
  home-to-partner displacement are collected into the same layer whenever
  their operands are disjoint. The proxy mover is the reset-identified
  ancilla when preferred, otherwise the first operand.
- Each displacement bucket is edge-colored into disjoint matchings, preserving
  repeated occurrences. Bipartite graphs use optimal edge coloring; other
  graphs use greedy coloring. Whole matchings are packed largest-first into
  compatible layers without splitting a matching; ties keep first-appearance
  order. This layer-packing heuristic is not a global pulse-count optimum.
- `reorder_cz=true` always enables this pass. The compatibility option
  `reorder_cz_by_displacement=true` also enables it independently.
- The pass in `dag_helper.py` preserves operation occurrences, quantum and
  classical wire dependencies, and barrier order on each barrier's qubits
  (a barrier is placed in the 1Q context and fences only its qubits). It rebuilds the
  single-qubit context and checked reference with the explicit layers.
  No later ASAP extraction may split equal-displacement bundles.
- This is logical reordering only: no TSP travel search, side selection,
  AOD grouping, movement emission, or load-retention planning occurs in the
  pass. Those decisions remain in ordinary `dynamics.py` scheduling.
- Layers contain disjoint operands, not entire overlapping CZ blocks.
  AOD span, alignment, and reuse settings may split a layer into
  several physical groups. Layer count and physical duration may increase;
  no original-layer or separate block-tour schedule is compared.
- The Python loader's geometry-free `reorder_cz` API remains available for
  callers without placement, but the CLI uses only the post-placement pass.
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
- A finite positive value permits an extra loaded atom only when its next use is known within the horizon.
- `Inf` permits indefinite eligibility but does not override legality or cost comparison.
- `T_reuse=0` permits identical-load continuation but forbids retaining extra loaded atoms.
- A larger horizon does not guarantee a monotonically shorter result because the reuse planner is greedy.

### 8.4 Configuration-controlled scheduling

`naive_n_dag.schedule_circuit` builds up to two complete layered schedules from
the same placement, validates and times each with the common model, and
returns the cheaper (`reuse` on ties):

- `reuse`: the layered planner with the configured AOD size and `T_reuse`.
- `individual` (unless `max_dimension` is already `[1, 1]`): the same layered
  planner with a `[1, 1]` AOD, one atom per load. A run therefore considers
  every candidate a one-atom-AOD run does, so it is never slower than moving
  one atom at a time.

Within the layered planner:

- Mover look-ahead: unless an ancilla is preferred, each gate moves the operand
  needed again soonest, and groups whose movers are needed soonest run last in
  the layer, so their load can be retained.
- Each layer also tries the one-atom-per-load partition and,
  for retained loads, a single reused gate; the cheaper closed cost is kept.

- `T_reuse` controls legal, cost-tested load retention, including identical-load
  continuation at zero. Returning home when necessary remains part of execution.
- The side-priced TSP tour orders unique displacements (Section 8.2). Only a
  failed tour triggers a warning and alignment ordering; there is no
  group-ordering configuration.
- The side DP always chooses interaction sides. The greedy side ordering was
  removed, and a config setting `_side_ordering` is rejected with an error.
- `selected` names the returned candidate; `candidate_times` lists every built
  candidate's duration.
- Replay validation and common-model timing remain mandatory for every
  candidate. Planning or validation errors propagate rather than silently
  dropping a candidate.

No no-reuse, alignment-only, or `naive_dag` baseline is built; `individual`
is the one-atom-at-a-time comparison.
Concorde is exact for its reduced displacement objective (fast_tsp is a
heuristic), not the complete physical duration; there is no duration guarantee against uncomputed policies.

With placement-aware preprocessing enabled, `main.py` passes only the reordered
layers to `schedule_circuit`. Otherwise it passes the original extracted layers.
There is one `schedule_circuit` call and no preprocessing-candidate duration comparison.
Legal CZ permutations and occurrence counts are checked before physical replay.

## 9. Serialization rules

- `T=0` contains the true initialization.
- Subsequent printed timestep numbers are consecutive.
- Every nonempty printed timestep contains either a gate batch or one movement batch.
- Movement labels are determined by endpoint parity: SLM-to-AOD is `load`, AOD-to-SLM is `unload`, and AOD-to-AOD is `move`.
- A qubit appears at most once in one movement batch.
- `naive_dag` emits one moving atom per movement timestep.
- `naive_n_dag` batches ordinary movements only when they share the required rigid displacement and action kind. Load and unload boundaries remain explicit.
- The header records `lattice_spacing`; `naive_n_dag` also records the `rydberg_radius` used for pulse separation.
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
- For `naive_n_dag`, no two simultaneous two-qubit gates with atoms strictly inside the Rydberg radius of each other (Section 5.1).
- Circuit identities, parameters, multiplicities, measurement destinations, and per-qubit order against the SWAP-filtered reference.
- No emitted SWAPs.
- No atom remaining active in the AOD at completion.
- The solver-specific terminal rule: changed valid homes are allowed for standalone `naive_dag`; exact original-home restoration is required for `naive_n_dag` and its accepted adapted candidates.
- Agreement between internal events, serialized actions, and reported duration.

## 11. Scope limitations

These rules define the repository's current discrete scheduling model. They do not claim to model arbitrary QASM control flow, finite optical-trap extent, crosstalk beyond the simultaneous-gate Rydberg-radius rule of Section 5.1 (for example, idle atoms near a pulsed pair), analog trajectory intersections away from represented grid points, hardware-specific measurement duration, or finite AOD travel bounds. Such constraints require separate rules and validation before being treated as hardware guarantees.
