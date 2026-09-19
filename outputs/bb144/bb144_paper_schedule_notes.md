# BB144 schedule derived from the paper

The generated `bb144_paper_optimized.schedule.txt` preserves the supplied QASM
and passes the current `naive_n_dag` replay validator. Its complete duration is
**6.353444 ms under the existing solver configuration**. It does **not** reach
2.97 ms under that configuration.

The same emitted moves take **2.834652 ms under the paper's Equation (3)**.
Adding the existing solver's gate/switch and transfer overheads gives
**2.924732 ms**, within 1.6% of 2.97 ms. This is a separately labeled timing
comparison, not a reproduction of the paper's circuit or an unchanged-solver
2.97 ms result. The schedule header always uses the actual solver timer.

| Schedule/model | Complete time |
| --- | ---: |
| Fresh current `naive_n_dag` run, same input/configuration/placement | 11.564650 ms |
| Separate X/Z collective loads, input wire order preserved | 7.871299 ms |
| Combined check load, input wire order preserved, solver timer | 6.353444 ms |
| Same combined schedule, paper Eq. (3) motion + solver overheads | 2.924732 ms |

The combined schedule is 45.1% shorter than the freshly generated solver
baseline. An older existing `bb144_n_init.schedule.txt` records 19.025968 ms;
the fresh validated baseline above is the appropriate current comparison.

## What the paper contributes

Source: `../../Matching_Generalized.pdf`, Sections V-B through V-E, Figure 5,
Figure 6, and Tables I–III (PDF pages 5–8).

The authors group interactions by their relative check/data displacement,
including periodic wraparound offsets, and route whole check arrays through
these stops. Section V-D says they use **Concorde**, a traveling-salesman solver;
it does not describe the result as a hand-solved optimum.

This implementation derives displacement groups directly from the input QASM.
A dynamic program orders their visits subject to the input's per-qubit
dependencies and chooses among four cardinal interaction sides. Paths use
the repository's legal AOD highways. Two candidates are compared: separate
loads of the X/Z check arrays, and one combined load of all 144 check atoms.
The combined candidate wins. Checks of the two types may interleave only
where input wire dependencies permit; intervening H/Z gates are preserved.

The selected schedule loads q[144] through q[287] once by displacement (0,-1),
keeps all data atoms in their SLM traps, performs 24 CZ batches, and returns
the complete load to its original homes. The full movement/gate sequence is
in the schedule; `bb144_paper_optimized.schedule.timeline.csv` gives all 66
timesteps, batch sizes, displacements, and cumulative times under both models.

The dynamic program minimizes motion within these specified collective-load,
one-visit-per-group candidates. It is not a proof of global scheduling
optimality; gate overhead is evaluated after route selection.

## Why the clocks differ

Both models use 5 micrometers between adjacent SLM sites. The JSON coordinates
are trap indices: the existing `initial_layout_fill` doubles them, and each
solver grid step is 2.5 micrometers. This conversion preserves physical spacing.
The JSON resembles the interleaved layout in Figure 5a, rather than explicitly
encoding the displaced, collision-free arrangement in Figure 5b.

The current configuration uses acceleration 2,750 m/s², velocity capped at
0.55 m/s, and a trapezoidal/triangular start-and-stop profile for every segment.
The paper's Section V-E instead uses
`sqrt(6*abs(dx_um)/0.02) + sqrt(6*abs(dy_um)/0.02)` microseconds, with an
acceleration parameter of 0.02 micrometers/microsecond² = 20,000 m/s².
That expression has no explicit velocity cap. The paper labels Table II as
movement costs; Table III repeats 2.97 ms for the localized-control case.

The combined schedule's solver costs are:

| Component | Time |
| --- | ---: |
| Motion, including one-step load/unload movement | 6,263.363611 µs |
| Two transfer surcharges | 30.000000 µs |
| 24 CZ pulses and 11 one-qubit pulses, including switches | 60.080000 µs |
| Total | 6,353.443611 µs |

The timeline's paper clock substitutes Equation (3) only for motion and
retains those 90.08 µs of solver overhead. It is a hybrid comparison explicitly
identified in the column name, not a second claim about the current hardware.

## The supplied QASM is not the paper's full check round

The copies in `../inputs/qasm_files/` and `../../qasm_outputs/` are identical
(SHA-256 `9874bfd8a3d30a4c6594f73f779c84d7ba9c42b94253c5e0123608f4d20a67f3`).
The root coordinates and the copy in `../inputs/qasm_files/` also match.

The QASM contains 864 CZ, 144 H, and 432 Z gates, with no measurement or reset.
Its generator constructs the X-labeled neighborhoods from A and its transpose,
and the Z-labeled neighborhoods from B and its transpose. This differs from
the paper's `Hx = [A|B]`, `Hz = [B^T|A^T]` construction. If the supplied
neighborhoods are interpreted as X and Z CSS stabilizers, 1,296 X/Z pairs
overlap on an odd number of data qubits and would anticommute.

Also, q[216] through q[287] have no H preparation/basis rotation: when they
start in |0>, their CZ gates cannot accumulate the intended check information.
The file therefore cannot itself be treated as a complete stabilizer
measurement round. No input circuit, placement, solver rule, or hardware
parameter has been modified to hide these differences.

## Validation and reproduction

Validation replays both the internal events and the written text, checking
rigid active-load ownership, legal highways/transfers, unique endpoints,
interaction separation, gate identities and per-qubit order, and final homes.
Serialized timestep boundaries and header duration are also checked. A direct
text comparison independent of the Qiskit DAG verifies all 1,440 operations
and all 288 per-qubit gate sequences against the original QASM.

These are the repository's discrete checks. They do not establish physical
crosstalk safety or analog path clearance beyond that model.

From `/home/gage/Quantum_Class`:

```bash
Phys765_Test/bin/python Phys765/bb144_paper_schedule.py
```

The generator writes the optimized schedule, timing/validation JSON, and CSV
timeline. The separate-load schedule and regenerated baseline are also retained
in this directory for comparison. Pre-existing inputs, schedules, and solver
source files were left unchanged.

## Acceleration and inactive-velocity-limit test

The separate configuration `../inputs/algorithms/bb144_n_paper_acceleration.json`
changes only `max_acceleration` to `20000 m/s^2` and `max_velocity` to `10 m/s`.
The latter is an effectively inactive limit for this test: the longest emitted
segment is 190 µm, with a triangular-profile peak speed of only 1.949359 m/s.
Raising the limit further to 1,000 m/s gives the same duration.

| Parameters applied to the optimized schedule | Complete solver time |
| --- | ---: |
| Original: 2,750 m/s², 0.55 m/s limit | 6.353444 ms |
| Paper acceleration: 20,000 m/s², original 0.55 m/s limit | 3.063958 ms |
| Paper acceleration: 20,000 m/s², inactive 10 m/s limit | **2.404564 ms** |

Reoptimizing with the new configuration selects the same event sequence.
`bb144_paper_acceleration.schedule.txt` passes internal and serialized replay
and direct input gate-trace validation. Its motion costs 2.314484 ms, with
0.030000 ms of transfer charges and 0.060080 ms of gate/switch charges.

This matches the paper's acceleration parameter but still uses the solver's
triangular motion profile: `2*sqrt(distance/a)` rather than the paper's
`sqrt(6*distance/a)`. Thus motion in this uncapped solver test is
`sqrt(2/3)` times the Equation (3) estimate for these same segments. The
2.404564 ms result is not an exact reproduction of the paper's motion model.

Reproduce from `/home/gage/Quantum_Class`:

```bash
Phys765_Test/bin/python Phys765/bb144_paper_schedule.py \
  --config Phys765/inputs/algorithms/bb144_n_paper_acceleration.json \
  --output Phys765/outputs/bb144_paper_acceleration.schedule.txt
```
