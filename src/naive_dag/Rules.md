# Scheduling Rules

This file sets out the problem every scheduler similar to `src/naive_dag` must solve and the
rules a valid schedule must follow. It does not depend on any solver. The
`Scheduler.md` next to this file explains how this folder's solver meets these
rules, including the extra restrictions it adds to avoid collisions.

## 1. The problem

The input is a quantum circuit (a QASM file) and a neutral-atom device. Each
logical qubit is one atom. The scheduler must produce a timed list of actions
that runs every operation of the circuit on the device:

- where each atom starts (the initial placement),
- when atoms are picked up, moved, and put down,
- when each gate and measurement runs.

The goal is a **short total schedule time** that obeys every rule below and if possible minimize overall movement and unload/loading of atoms.

## 2. The device

- **SLM traps**: a fixed rectangular grid (for this instance) of static traps.
  Atoms rest here, and an atom in an SLM trap cannot move. `lattice_spacing`
  (config) is the distance between neighbouring traps.
- **Highways**: the lanes between the traps. Moving atoms travel along them.
- **AOD tweezers**: movable tweezers. An atom must be transferred into the AOD
  before it can move, and transferred back to an SLM trap afterwards. The AOD
  may hold more than one atom at a time.

## 3. Positions and collisions

- Every atom has exactly one known position at every moment of the schedule,
  whether it is in an SLM trap or in the AOD. The scheduler must keep this
  bookkeeping exact, including for every atom in a multi-atom AOD load.
- At most one atom may occupy any spot at any time.
- Atoms may not collide, either at the end of a movement or while moving.

How a solver places atoms and plans moves to avoid collisions is up to that
solver (see `Scheduler.md`).

## 4. Moving atoms

There are three kinds of movement action:

| Action | From | To |
|---|---|---|
| `load` | SLM trap | AOD |
| `move` | AOD | AOD, along the highways |
| `unload` | AOD | SLM trap |

- An atom may go directly from one SLM trap to another as long as no
  collision happens and no spot ever holds more than one atom.
- An `unload` target must be empty when the unload happens. It may have been
  occupied earlier.
- A movement of length zero is not an action and takes no time.
- Each atom's next action starts where its previous action ended.
- At the end of the schedule no atom may be left in the AOD.

### 4.1 How the AOD moves

The AOD is a crossed set of **row tones** and **column tones**. A tweezer
sits at every row-column intersection, and every atom in the AOD sits on one
row and one column. A movement shifts rows vertically and columns
horizontally. In one movement the AOD may:

- **translate**: every row and column shifts by the same amount, so all loaded
  atoms move with one common displacement;
- **stretch or contract**: rows shift by different amounts (or columns do),
  changing the gaps between them;
- do any mix of the two,

as long as, for rows and for columns separately:

1. **No crossing.** Rows keep their top-to-bottom order and columns keep
   their left-to-right order. A row may not pass another row.
2. **No merging.** Two rows that start on different lines end on different
   lines. The same holds for columns.
3. **No splitting.** Atoms on one row are on one tone, so they end on one row.
   The same holds for columns.
4. **Acceleration limit.** No tone accelerates faster than `max_acceleration`
   (Section 7.1).

Every atom in the AOD is on some row and column, so it moves whenever its
tones move. An AOD atom that does not change position during a movement
still counts, with zero displacement. Rules 1-3 come from the ordering
constraint described in the Enola and PowerMove papers: breaking it can make
atoms collide and be lost.

### 4.2 Finding row/column crossings

**Pairwise sign test.** For every pair of atoms `a`, `b` in the AOD during
one movement batch:

```text
sign(row_a - row_b) at the start == sign(row_a - row_b) at the end
sign(col_a - col_b) at the start == sign(col_a - col_b) at the end
```

One test covers all three violations:

| Start sign | End sign | Violation |
|---|---|---|
| `+` | `-` (or `-` to `+`) | crossing |
| `+` or `-` | `0` | merging |
| `0` | `+` or `-` | splitting |

Practical notes:

- **Use the sign test, not a one-sided inequality.** PowerMove writes the
  x-conflict as `x1_start <= x2_start` but `x1_end > x2_end` (or the mirror
  case). Read literally, that misses merging (different starts, equal ends),
  although its Figure 5 shows merging as a conflict. Enola lists merging
  explicitly.
- **Sort instead of comparing every pair.** Group the loaded atoms by starting
  row. Each group must have exactly one ending row, and the groups, sorted by
  starting row, must have strictly increasing ending rows. Repeat for columns.
  This takes `O(n log n)` time instead of `O(n^2)`.
- **Endpoints are enough when all tones share one time profile.** If every
  tone in a batch follows `position(t) = start + displacement * s(t)` with the
  same `s(t)`, the difference between two tones is linear in `s`, so a sign
  that holds at both ends holds throughout. If tones use different profiles
  (for example, each finishing its own trapezoid at a different time), a
  faster tone can overtake a slower one mid-move, so check intermediate times
  too.
- **Check each batch on its own**, starting from the positions where the
  previous batch ended. Loads and unloads are movement batches too.
- **A rigid batch never violates these rules.** If every loaded atom has the
  same displacement, no order changes.
- **A one-atom AOD never violates them** either, since a single atom has no
  pairs.

## 5. Gates

### 5.1 Two-qubit gates

The Rydberg radius is `RYDBERG_RADIUS = 15 micrometers`. A two-qubit gate can
run only when its two atoms are within the Rydberg radius of each other. The
rules do not say which tweezer each atom must be in.

### 5.2 Simultaneous two-qubit gates

Two-qubit gates may share one pulse only when every atom of each gate is at
least `RYDBERG_RADIUS` from every atom of every other gate in that pulse.
Atoms that are not part of the pulse are not checked. Gates that would
interfere must run in separate pulses, and each pulse is charged its own time
(Section 7.3).

### 5.3 Single-qubit gates and measurements

Single-qubit gates and measurements can be applied to an atom wherever it is,
loaded or not. They keep their place in the circuit order.

### 5.4 Gates in the same timestep

- A gate may not use the same qubit twice.
- Gates in the same timestep must act on different qubits.

## 6. The circuit must be preserved

The schedule must run the same circuit as the input, up to the reorderings
allowed below.

- **SWAPs are removed** from the input before scheduling. No SWAP is emitted,
  and the scheduler may not add SWAPs, routing SWAPs, or SWAP decompositions.
  Atoms are moved physically instead.
- Every other supported operation appears **exactly once**: nothing added,
  repeated, or dropped.
- Gate names, parameters, and measurement destinations (`measure q[2] ->
  c_x[5];`) are kept.
- Operations on each qubit keep their order. Operations on different qubits
  may be reordered or run together.
- A QASM barrier is a scheduling boundary, not a physical action. Operations
  on its qubits may not cross it. Each solver's `Scheduler.md` says exactly
  how it applies barriers.
- Only one- and two-qubit operations are supported. Larger operations must be
  rejected, never scheduled silently.

## 7. Timing

The schedule time is the sum of every movement, transfer, gate, and
measurement charge.

### 7.1 Movement

Every emitted movement starts and stops at rest. `movement_shape` picks the
motion profile. Here `distance` is the physical straight-line length of the
movement, and `delta_x`, `delta_y` are its physical lengths along the rows and
columns.

**`trapezoidal`** (velocity `v = max_velocity`, acceleration
`a = max_acceleration`):

```text
ramp_distance = v^2 / a

if distance > ramp_distance:
    time = 2v/a + (distance - ramp_distance)/v
else:
    time = 2 * sqrt(distance/a)
```

**`Cubic_Spline`** (minimum-jerk; does not use `max_velocity`):

```text
time = sqrt(6 * delta_x / a) + sqrt(6 * delta_y / a)
```

- Atoms moved together in one batch cost the time of the longest move in the
  batch, not the sum.
- Consecutive same-direction pieces emitted as one segment are timed as one
  segment.
- A turn, gate, load, unload, or reversal is a stop/start boundary.

### 7.2 Transfers

- Each load or unload batch costs `transfer_SLM_AOD`, once per batch.
- Ordinary highway moves have no transfer cost.

### 7.3 Gates and measurements

| Batch | Cost |
|---|---|
| One single-qubit pulse | `average_single_gate_time + t_switch` |
| One two-qubit pulse | `average_two_gate_time + t_switch` |
| All measurements in one timestep | `measurement_time + t_switch` |

- A single-qubit pulse is identified by gate name plus parameters. The same
  pulse on several atoms in one timestep is charged once. Different pulses
  in one timestep are charged separately.
- Every separate two-qubit pulse is charged, including pulses split apart by
  the Rydberg rule (Section 5.2).
- `measurement_time` defaults to `0 milliseconds`.

Two schedules may only be compared when they use the same timing model and
the same initial placement.

## 8. What is not modelled

Finite trap size, laser crosstalk beyond the simultaneous-gate Rydberg rule
(for example, idle atoms near a pulsed pair), AOD travel limits,
"ghost" tweezers at unused row-column intersections, several independent
AODs, hardware-specific measurement time, and QASM control flow are not
modelled.
These would need their own rules and checks before being treated as hardware
guarantees.
