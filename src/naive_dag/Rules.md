# Scheduling Rules

This file sets out the problem every scheduler similar to `src/naive_dag` must solve and the
rules a valid schedule must follow. It does not depend on any solver: anyone
may write their own scheduler for it. The `Scheduler_ND.md` next to this file explains how
this folder's solver meets these rules, including the extra restrictions
(heuristics) it adds that the rules do not require.

## 1. The problem

The input is a quantum circuit (a QASM file) and a neutral-atom device. Each
logical qubit is one atom. The scheduler must produce a timed list of actions
that runs the circuit on the device:

- where each atom starts (the initial placement),
- when atoms are picked up, moved, and put down,
- when each gate and measurement runs.

The goal is a **short total schedule time** that obeys every rule below and, if
possible, also keeps the overall movement and the number of loads and unloads
small.

## 2. The device

All lengths and times below are config parameters; a schedule states the
values it uses.

- **SLM traps**: a fixed rectangular grid (for this instance) of static traps.
  Atoms rest here, and an atom in an SLM trap cannot move. `lattice_spacing`
  is the distance between neighbouring traps.
- **Highways**: the lanes between the traps.
- **AOD tweezers**: movable tweezers. An atom must be transferred into the AOD
  before it can move, and transferred back to an SLM trap afterwards. The AOD
  may hold more than one atom at a time.
- **AOD span**: `max_dimension = [rows, cols]`. All atoms held in the AOD at
  the same time must fit in a window `rows` traps tall and `cols` traps wide:
  their positions may span at most `(rows - 1) * lattice_spacing` vertically
  and `(cols - 1) * lattice_spacing` horizontally. Any smaller load is allowed.
- **Trap size**: `trap_radius` (default 1 micrometer) is the radius of the
  potential well of every trap and tweezer (Section 3).

## 3. Positions and collisions

- Every atom has exactly one known position at every moment of the schedule,
  whether it is in an SLM trap or in the AOD. The scheduler must keep this
  bookkeeping exact, including for every atom in a multi-atom AOD load.
- **No overlapping traps.** Each atom sits in a trap of radius `trap_radius`
  centred on it. The traps of two different atoms may never overlap, so their
  centres must stay at least `2 * trap_radius` apart **at every moment**, at
  the ends of a movement and all along it. Checking the start and end of a
  movement is not enough unless the motion makes that sufficient (Section
  4.2); otherwise check the closest approach along the actual trajectories.
- An atom moving midway between two rows of traps is `lattice_spacing / 2` from
  them, so the geometry needs `lattice_spacing >= 4 * trap_radius` (4
  micrometers at the default). Moving atoms that close is risky; 8 micrometers
  is the suggested minimum, and experimentalists may tune it.

How a solver places atoms and plans moves to avoid collisions is up to that
solver (see `Scheduler_ND.md`).

## 4. Moving atoms

There are three kinds of movement action:

| Action | From | To |
|---|---|---|
| `load` | SLM trap | AOD |
| `move` | AOD | AOD |
| `unload` | AOD | SLM trap |

- An atom may go from one SLM trap to another directly (load, move, unload)
  as long as no traps overlap (Section 3). Its cost is the two transfers plus
  the movement (Section 7).
- An `unload` target must be empty when the unload happens. It may have been
  occupied earlier.
- A movement of length zero is not an action and takes no time.
- Each atom's next action starts where its previous action ended.
- Atoms need not return to their starting traps.
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
4. **Motion limits.** No atom exceeds the limits of Section 7.1, and the AOD
   span (Section 2) holds throughout.

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

These tests cover the AOD's own atoms. Atoms resting in SLM traps that a moving
atom passes must be checked separately with Section 3.

## 5. Gates

### 5.1 Two-qubit gates

The Rydberg radius is `RYDBERG_RADIUS = 15 micrometers`. A two-qubit gate can
run only when its two atoms are **strictly closer** than the Rydberg radius.
Which tweezer each atom is in does not matter: both may be in SLM traps, both
in the AOD, or one of each. With traps 15 micrometers apart, neighbouring
traps are exactly at the radius, so at least one atom must move; with traps 8
micrometers apart, side and diagonal (corner) neighbours can be gated where
they stand.

Gate time and fidelity depend on the distance between the two atoms. The
current timing model charges one `average_two_gate_time` for every
two-qubit pulse (Section 7.3); a schedule that gates atoms at different
distances should report those distances so the difference can be accounted
for.

### 5.2 Simultaneous two-qubit gates

Two-qubit gates may share one pulse only when every atom of each gate is at
least `RYDBERG_RADIUS` from every atom of every other gate in that pulse.
Atoms that are not part of the pulse are not checked. Gates that would
interfere must run in separate pulses, and each pulse is charged its own time
(Section 7.3).

### 5.3 Single-qubit gates, measurements and resets

- Single-qubit gates and measurements can be applied to an atom wherever it
  is, loaded or not. They keep their place in the circuit order.
- **A reset needs the atom in an SLM trap**: unload it first.

### 5.4 Gates in the same timestep

- A gate may not use the same qubit twice.
- Gates in the same timestep must act on different qubits.

### 5.5 Ancillas

Ancillas are the qubits of the QASM registers named in `ancilla_registers`.
When `prefer_moving_ancillas` is set, every two-qubit gate with exactly one
ancilla must move the ancilla, not the data atom (ancillas can be reset, so
moving them costs less fidelity). It is recommended whenever the circuit
declares ancilla registers; with it unset, either atom may move.

## 6. The circuit must be preserved

The schedule must give **the same measurement statistics as the input
circuit**, up to a stated tolerance. Within that:

- **Preprocessing is allowed.** The circuit may be transpiled and optimized
  before scheduling (for example Qiskit at optimization level 2, which deletes
  rotations too small to matter and phase-only gates directly before a
  measurement). The settings must be recorded with the schedule. Preprocessing
  must respect barriers.
- **SWAPs are gates.** A `swap` in the input is an ordinary two-qubit gate and
  is scheduled like one; so are gates such as `iswap`. Removing a SWAP is a
  preprocessing choice left to the user, for example by editing the input or
  by transpiling (Qiskit at level 2 removes SWAPs and rewires the
  measurements). A SWAP at the end of a circuit, such as the QFT bit reversal,
  can also be undone in data analysis. A scheduler may not add SWAPs.
- **After preprocessing, every operation runs exactly once**: nothing added,
  repeated, or dropped.
- Gate names, parameters, and measurement destinations (`measure q[2] ->
  c_x[5];`) are kept: each classical bit receives the state the QASM measures
  into it.
- Operations on each qubit keep their order, except that operations which
  commute (for example CZ gates sharing a qubit) may be reordered. Operations
  on different qubits may be reordered or run together.
- **Barriers.** A QASM barrier is a scheduling boundary, not a physical action.
  For the qubits it names, every operation before the barrier must run in a
  strictly earlier timestep than every operation after it. A solver may be
  stricter (for example by treating every barrier as covering all qubits).
- Only one- and two-qubit operations are supported. Larger operations must be
  rejected, never scheduled silently.

## 7. Timing

The schedule time is the sum of every movement, transfer, gate, and
measurement charge.

### 7.1 Movement

Any motion profile is allowed as long as it is physically consistent:

- every movement starts and ends with its atoms at rest;
- no atom's acceleration ever exceeds `max_acceleration`, and, if the profile
  uses a velocity limit, no atom's speed exceeds `max_velocity`;
- a batch of atoms moved together lasts until its last atom stops; batches are
  charged that time, not the sum of their atoms' moves;
- the trajectories that are timed are the ones checked for collisions
  (Section 3) and crossings (Section 4.2).

With one shared time profile for all tones of a batch, the atom with the
largest displacement sets the time. The displacement of an atom is its row
tone's shift combined with its column tone's shift, so timing the batch over
`hypot(largest row shift, largest column shift)` is always safe.

Two reference profiles, for one straight movement of length `distance` (or
`delta_x`, `delta_y` along the rows and columns), with
`a = max_acceleration` and `v = max_velocity`:

**`trapezoidal`** (constant acceleration to `v`, cruise, constant
deceleration):

```text
ramp_distance = v^2 / a

if distance > ramp_distance:
    time = 2v/a + (distance - ramp_distance)/v
else:
    time = 2 * sqrt(distance/a)
```

**`Cubic_Spline`** (cubic position profile, peak acceleration `a` at the start
and end; no velocity limit):

```text
time = sqrt(6 * distance / a)              # one straight movement
time = sqrt(6 * delta_x / a) + sqrt(6 * delta_y / a)   # axes one after the other
```

Other profiles may be used and added to configs, as long as they meet the
limits above.

- Consecutive same-direction pieces emitted as one segment are timed as one
  segment.
- A turn, gate, load, unload, or reversal is a stop/start boundary.

### 7.2 Transfers

- Each handoff between a static trap and a tweezer (a `load` or an `unload`)
  is a transfer. All handoffs in the same timestep are charged one
  `transfer_SLM_AOD` together.
- A direct trap-to-trap move therefore costs a load transfer, its movement,
  and an unload transfer.
- Steps in which atoms only stay held in the AOD, with no handoff, have no
  transfer cost. Ordinary moves have no transfer cost.

### 7.3 Gates, measurements and resets

| Batch | Cost |
|---|---|
| One single-qubit pulse | `average_single_gate_time + t_switch` |
| One two-qubit pulse | `average_two_gate_time + t_switch` |
| All measurements in one timestep | `measurement_time + t_switch` |
| A reset | no cost yet |

- A single-qubit pulse is identified by gate name plus parameters. The same
  pulse on several atoms in one timestep is charged once. Different pulses
  in one timestep are charged separately.
- Every separate two-qubit pulse is charged, including pulses split apart by
  the Rydberg rule (Section 5.2).
- `measurement_time` defaults to `0 milliseconds`.
- A reset is a placeholder: a real reset takes several steps, which a future
  version of these rules will cost.

### 7.4 Comparing schedules

Two schedules may be compared when they use the same device and timing
parameters (Sections 2 and 7) and obey these rules. They may use different
initial placements, strategies, and motion profiles that meet Section 7.1; any
difference in the timing model must be stated.

## 8. What is not modelled

Laser crosstalk beyond the simultaneous-gate Rydberg rule (for example, idle
atoms near a pulsed pair), distance-dependent gate time and fidelity (Section
5.1), the time a reset takes, AOD travel limits beyond the span, "ghost"
tweezers at unused row-column intersections, several independent AODs,
hardware-specific measurement time, and QASM control flow are not modelled.
These would need their own rules and checks before being treated as hardware
guarantees.
