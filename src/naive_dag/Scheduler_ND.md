# `naive_dag` Scheduler

How the `naive_dag` solver schedules a circuit: its coordinate model,
circuit preservation, state tracking, movement and timing rules, the
one-moving-atom algorithm, serialization, and validation. The
solver-independent problem rules are summarised in [Rules.md](Rules.md).

Implementation references:

- [Dynamics and scheduling](dynamics.py)
- [DAG preparation](dag_helper.py)

## 1. Coordinate model

`Rules.md` only requires a fixed rectangular grid of SLM traps with highways
between them. This solver adds the parity model below as a collision-avoidance
heuristic: traps sit only on even-even sites and moving atoms travel only along
odd lanes, so a moving atom never passes through a trap.

Positions are integer grid coordinates `(row, column)`. One grid step is
`lattice_spacing / 2`.

- An **SLM trap** is an even-even coordinate: both row and column are even.
- Every other coordinate is an **AOD position**.
- Horizontal AOD highway movement is legal only on an odd row.
- Vertical AOD highway movement is legal only on an odd column.
- An odd-odd coordinate is a highway intersection where movement may change axis.
- For configured SLM dimensions `[R, C]`, initial and unloading trap coordinates lie within rows `0..2R-2` and columns `0..2C-2`.
- AOD highway coordinates may extend outside that rectangle. The current model does not impose a finite physical AOD travel boundary.

```text
column:   0 1 2 3 4
row 0:    S | S | S      S  SLM trap (even, even)
row 1:    - + - + -      -  horizontal highway (odd row)
row 2:    S | S | S      |  vertical highway (odd column)
row 3:    - + - + -      +  intersection (odd, odd)
row 4:    S | S | S
```

A straight movement segment is judged by its complete interior, not only its endpoints. A segment may not pass through an even-even SLM coordinate. For example, `(6,1) -> (6,5)` is illegal because horizontal movement is on even row 6 and crosses SLM traps.

## 2. Circuit preservation

- SWAP operations are ordinary two-qubit gates: they are scheduled and emitted like any other gate (`swap q[a],q[b];`). The solver never removes or adds SWAPs; removing them is left to the user (edit the QASM, or transpile: Qiskit at levels 2-3 removes SWAPs and rewires the measurements).
- Ancilla status (`ancilla_registers`) is a property of the atom: the qubits of the named QASM registers.
- Optional Qiskit transpile at load time: `transpile_optimization_level` (0-3, default 0) and `transpile_basis_gates` (a list of gate names, default `null` = gates as written). Level 0 with no basis skips Qiskit entirely, so the circuit is kept exactly as written. Otherwise the transpiled circuit replaces the input for scheduling and for every check below, and the header records `transpile: optimization_level=... basis_gates=...`. Levels 2-3 may delete gates Qiskit judges equal to the identity (QFT-63: controlled phases below about pi/2^19, 3400 -> 2014 two-qubit gates), so the schedule then preserves the transpiled circuit, not the file.
- Circuit comparison is performed against the prepared operation sequence (after the optional transpile).
- Every remaining supported one- and two-qubit operation must appear exactly once. No operation may be added, duplicated, or omitted.
- Gate identity, parameters, measurement destinations, and per-qubit operation order must be preserved.
- Operations on disjoint qubits may be reordered or emitted in the same timestep when the scheduler's batching rules permit it.
- A gate may not use the same qubit twice.
- Simultaneous gates must have disjoint operands.
- Measurements retain the classical register and destination, for example
  `measure q[2] -> c_x[5];`. Equal local indices in different registers remain distinct.
- A QASM may declare several quantum registers. The loader numbers their qubits
  globally in declaration order (`qreg data[144]; qreg ax[72];` gives `ax[i] =
  q[144 + i]`), and the schedule writes `q[...]` with a `qubit_registers:`
  header line recording the mapping. Classical registers are kept as written.
- Barriers are scheduling metadata and are not emitted as physical operations.
  When the DAG is built every barrier is widened to all qubits, so each
  barrier is a global boundary: every operation before it is emitted before
  every operation after it, whichever qubits the barrier named.
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

The one-step transfers and parity highways below are this solver's choices
for avoiding collisions; `Rules.md` is less strict.

At most one atom is ever in the AOD (Section 7.1), so the row/column ordering
rules of `Rules.md` Section 4.1 cannot be violated.

- A `load` starts at an even-even SLM trap and ends at a non-even-even AOD position.
- An `unload` starts at an AOD position and ends at an even-even SLM trap.
- Every load and unload is exactly one cardinal grid step.
- Direct SLM-to-SLM movement is never emitted. `Rules.md` allows it when no
  collision occurs; the solver avoids it so every move stays on a highway.
- An ordinary `move` has AOD positions at both endpoints.
- Ordinary movement is axis-aligned and follows the parity highways described in Section 1.
- Zero-length movement events are omitted and cost no time.
- A movement endpoint may not coincide with another atom.
- A highway route may not pass through an occupied SLM trap; the stronger parity rule normally excludes every SLM trap from an ordinary segment regardless of occupancy.
- An unload destination must be vacant at the moment of unloading. It is allowed to have been occupied earlier in the schedule.
- Every schedule ends with no atom left loaded in the AOD.

The current discrete model checks grid-position collisions and legal highway interiors. It does not model finite trap extent, analog collision envelopes between moving paths, laser crosstalk, or finite AOD travel boundaries.

## 5. Two-qubit interaction geometry

`Rules.md` only requires a gate's two atoms to be within the Rydberg radius.
This solver uses a stricter geometry, chosen to keep loaded atoms off the
traps and make collision checks simple. At every emitted two-qubit gate:

- Exactly one operand is in the AOD.
- Exactly one operand remains in an even-even SLM trap.
- The two operands are exactly one cardinal grid step apart, so their Manhattan distance is `1`.
- The loaded operand must be the atom tracked by the scheduler as active in the AOD.
- The solver never emits a gate with both operands loaded.
- The solver never emits a gate with both operands in SLM traps.

### 5.1 One two-qubit gate per pulse

Only one atom is ever in the AOD (Section 7.1), and every two-qubit gate
needs its AOD operand, so each two-qubit pulse contains exactly one gate.
The Rydberg-radius separation rule between simultaneous gates (`Rules.md`
Section 5.2) is therefore met automatically and is not checked separately.

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
  before a data-ancilla gate so that the ancilla (a qubit of `ancilla_registers`) can move.
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

## 8. Serialization rules

- `T=0` contains the true initialization.
- Subsequent printed timestep numbers are consecutive.
- Every nonempty printed timestep contains either a gate batch or one movement batch.
- Movement labels are determined by endpoint parity: SLM-to-AOD is `load`, AOD-to-SLM is `unload`, and AOD-to-AOD is `move`.
- A qubit appears at most once in one movement batch.
- One moving atom is emitted per movement timestep.
- The header records `lattice_spacing`.
- Serialized gate text contains all parameters and measurement destinations needed for independent replay.
- The serialized `final_time` must equal the duration recomputed from the serialized actions under the applicable common timing model.

## 9. Validation requirements

Before a complete schedule is accepted or written, it must be replayed from its true initialization and checked for:

- Initialization uniqueness, parity, and bounds.
- Movement continuity and correct action labels.
- Legal cardinal transfers and legal highway interiors.
- Active-load ownership: a load only into an empty AOD, and only the active atom moves or unloads (Section 7.1).
- Mandatory immediate reuse: an atom needed by the next two-qubit gate is not unloaded, except under `prefer_moving_ancillas` (Section 7.2).
- With `prefer_moving_ancillas=true`, the ancilla operand of each data-ancilla gate is the mover.
- Vacant movement endpoints and unload destinations.
- Correct two-qubit interaction geometry.
- Circuit identities, parameters, multiplicities, measurement destinations, and per-qubit order against the prepared reference.
- No operation crossing a QASM barrier (Section 2).
- No atom remaining active in the AOD at completion.
- The terminal rule: atoms may finish at changed valid homes (Section 7.3).
- Agreement between internal events, serialized actions, and reported duration.

## 10. Scope limitations

These rules define the repository's current discrete scheduling model. They do not claim to model arbitrary QASM control flow, finite optical-trap extent, crosstalk (for example, idle atoms near the pulsed pair), analog trajectory intersections away from represented grid points, hardware-specific measurement duration, or finite AOD travel bounds. Such constraints require separate rules and validation before being treated as hardware guarantees.
