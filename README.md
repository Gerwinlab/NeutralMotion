# Phys765

Phys765 is a neutral-atom circuit scheduling package. It loads QASM circuits, maps logical qubits to lattice sites, schedules gate/movement actions, and emits a timestep schedule text file.

## Install

Requires Python 3.10 or newer. From the repository root:

```bash
python3 -m pip install -e .
```

To also install the exact Concorde TSP solver (built from source; it is not on
PyPI), use the `concorde` extra:

```bash
python3 -m pip install -e ".[concorde]"
```

Without it, `naive_n_dag` warns once and solves displacement tours with
`fast-tsp`, which is always installed.

The notebooks in `Examples/` also need matplotlib, pylatexenc and pandas
(the `examples` extra); drawing DAGs additionally needs graphviz:

```bash
python3 -m pip install -e ".[examples]"
```

This installs the package in editable mode and exposes these CLI commands:

- `naive_dag`
- `naive_n_dag`

## Repository Layout

- `src/`: the `naive_dag` and `naive_n_dag` packages, each with its own
  `Rules.md` and scheduler doc (`Scheduler_ND.md`, `Scheduler_NND.md`; see
  [Scheduling Rules And Solver Docs](#scheduling-rules-and-solver-docs));
  `tests/`: the test suite.
- `inputs/qasm_files/`: every QASM circuit, grouped by family (`qLDPC/` with its
  `checks/` and `manifest.json`, `bb144/`, `qft/`, `bv/`, `vqe/`, `general/`).
- `benchmarks/configs/`: scheduler JSONs grouped by family. A config that uses
  an initial placement (`initial_layout`) or a `step_order` file lives in its
  own folder together with that file, e.g.
  `benchmarks/configs/bb144/bb144_n_interleaved/`. Config paths are relative to
  the config file.
- `benchmarks/output/`: schedules, FastSA logs, result CSVs (`results/`) and
  plots (`plots/`); `benchmarks/*.py` and `analysis_notebook.ipynb` are the
  benchmark scripts and analysis.
- `Examples/`: tutorial and example notebooks.
- `Legacy/`: the legacy cost F (`legacy_f.py`), the F vs F_time benchmark
  (`cost_fidelity.py`, `fastsa_tuning.py`) and old benchmark plots.

## Quick Run (`naive_n_dag`)

Example:

```bash
naive_n_dag benchmarks/configs/qft/qft_06_n_fastsa.json qft_6.qasm benchmarks/output --seed 0 --output-name qft_06_n_fastsa_example --log
```

This writes:

- `benchmarks/output/qft_06_n_fastsa_example.schedule.txt`
- `benchmarks/output/qft_06_n_fastsa_example.fastsa_log.csv` (because `--log` is enabled with FastSA)

QASM files may use several quantum and classical registers. Quantum registers
are numbered globally in declaration order (for `qreg l[63]; qreg r[63];`,
`r[0]` is `q[63]`); the schedule header line `qubit_registers:` records the
mapping, and measurements keep their classical register
(`measure q[126] -> c_x[0];`). `num_NA` counts the qubits of all registers.
Register names may not reuse `qelib1.inc` gate names such as `sx`.

## What The Schedulers Do

At a high level, `naive_dag` / `naive_n_dag`:

- Load QASM and convert to DAG-style gate ordering.
- Build a neutral-atom lattice from config dimensions and `lattice_spacing`.
- Place qubits on valid trap sites.
- Schedule movement + gate application over timesteps.
- Emit a readable schedule (`T=...`, `initialize`, `load/move/unload`, gate lines).
  An atom unloaded and then reloaded by the next transfer step stays in the
  AOD instead: only atoms that change traps get `load`/`unload`, and the kept
  atom appears as a `move` in the same step (it passes over its own home site).

QASM barriers are hard scheduling boundaries for the qubits they name. They
are not printed as physical actions. No gate or measurement on a barrier's
qubits may move from one side of it to the other, or share a timestep with
one from the other side; operations on other qubits may cross it. A barrier
on every qubit (`barrier q;`) is therefore a global boundary. `naive_dag`
widens every barrier to all qubits (see
[`src/naive_dag/Scheduler_ND.md`](src/naive_dag/Scheduler_ND.md)).

## Scheduling Rules And Solver Docs

Every solver folder in `src/` has two documents:

- `Rules.md`: the solver-independent problem: the device, collisions, legal
  AOD moves and how to check for row/column crossings, gate conditions
  (Rydberg radius), circuit preservation, and the timing model.
- `Scheduler_ND.md` (`naive_dag`) or `Scheduler_NND.md` (`naive_n_dag`): how that solver tackles it: its coordinate model and
  collision-avoidance heuristics, circuit preservation details, the
  scheduling algorithm, serialization, and validation checks.

| Solver | Rules | Scheduler |
|---|---|---|
| `naive_dag` | [`src/naive_dag/Rules.md`](src/naive_dag/Rules.md) | [`src/naive_dag/Scheduler_ND.md`](src/naive_dag/Scheduler_ND.md) |
| `naive_n_dag` | [`src/naive_n_dag/Rules.md`](src/naive_n_dag/Rules.md) | [`src/naive_n_dag/Scheduler_NND.md`](src/naive_n_dag/Scheduler_NND.md) |

## Config Notes

Common required timing keys in JSON configs:

- `average_single_gate_time`
- `average_two_gate_time`
- `t_switch`
- `transfer_SLM_AOD`
- `max_acceleration`
- `max_velocity`
- `lattice_spacing` (SLM trap pitch; one grid step is half of it)

`lattice_spacing` replaces the old `rydberg_radius` key, which is now rejected
with an error. The Rydberg radius itself is the constant `RYDBERG_RADIUS = 15
micrometers` in `src/naive_n_dag/dynamics.py`: `naive_n_dag` splits two-qubit
gates whose atoms lie strictly inside that radius of each other into separate
timesteps, each charged `average_two_gate_time + t_switch` (see
[`src/naive_n_dag/Scheduler_NND.md`](src/naive_n_dag/Scheduler_NND.md) Section 5.1).

`measurement_time` is optional and defaults to `0 milliseconds`. All
measurements emitted in the same timestep share one measurement-time charge,
plus one `t_switch`, regardless of how many qubits are measured in that step.

Scheduling behavior flags:

- `movement_shape`: selects the shuttle timing profile. `trapezoidal` uses the
  velocity-limited acceleration profile. `Cubic_Spline` (constant jerk) uses
  `sqrt(6 * delta_x / acceleration) + sqrt(6 * delta_y / acceleration)`, with
  physical axis distances in meters. `Quintic_Spline` (minimum jerk) uses the
  same form with `10 / sqrt(3)` in place of `6`.
- `ancilla_registers`: list of the QASM's ancilla register names, e.g.
  `["ax", "az"]`. Every qubit of those registers is an ancilla; a name the QASM
  does not declare is an error. The qLDPC QASMs declare `data`, `ax` and `az`
  (P126/P254: `l`, `r`, `ax`, `az`).
- `prefer_moving_ancillas`: optional boolean, default `false`. When `true`, a
  two-qubit gate with exactly one ancilla (from `ancilla_registers`, which is
  then required) always moves that ancilla instead of its data-qubit partner.
  If neither or both operands are ancillas, the normal mover heuristic is used.
- `transpile_optimization_level`: optional, `0`-`3`, default `0`. Runs the
  QASM through Qiskit's `transpile` at this level before anything else, in
  both solvers. Level 0 with no basis leaves the circuit as written. Levels 2-3
  can delete near-identity gates (QFT-63: 3400 -> 1554 two-qubit gates with
  Qiskit 2.5, 2014 with Qiskit 2.3), and the schedule then preserves the
  transpiled circuit.
- `transpile_basis_gates`: optional list of gate names, default `null` (gates
  as written). PowerMove uses `["u1", "u2", "u3", "cz", "id"]`. Settings in use
  are written to the schedule header as `transpile: ...`.
- SWAPs are scheduled as ordinary two-qubit gates (`swap q[a],q[b];`, charged
  one two-qubit pulse like any gate left as written). Whether a SWAP is needed
  is up to the user: remove it from the QASM, let the transpile remove it
  (Qiskit at levels 2-3 removes SWAPs and rewires the measurements), or set a
  CZ basis so it is charged as the CZs it decomposes into. End-of-circuit SWAPs
  such as the QFT bit reversal can also be undone in data analysis.
- `max_dimension`: the AOD size `[rows, cols]` in atoms. `naive_n_dag` always
  moves several atoms in one AOD load when this size (and `alignment_conc`)
  allow; set it to `[1, 1]` to move one atom at a time. The old `parallel` key
  was removed, and a config that still sets it is rejected with an error.

`naive_n_dag` orders the targets of every ordinary legal 2Q group, including CX
and other supported gates and with either value of the CZ reordering flags, by
a side-priced TSP tour, then fixes interaction sides with a fixed-order side
DP. The tour is solved by Concorde (`pyconcorde`); if pyconcorde is missing a
warning is issued and `fast_tsp` is used instead. Graphs with 4 or fewer nodes,
which Concorde cannot solve, always go to `fast_tsp`, whose local search is
bounded by `tsp_time_limit` (seconds per solve, default `0.05`). There is no
size limit. If Concorde fails (an error, or a tour that fails validation),
`fast_tsp` is tried next, with a warning. Only if no solver yields a valid tour
does `_sort_group_by_alignment` order the group instead, with a warning; the two methods are never both run and compared. Fresh groups optimize loading direction;
reused groups start at their actual AOD offset without reloading. Home return
is included in the cost estimate but deferred to allow reuse. Only disjoint
gates within a group are reordered; noncommuting wire dependencies remain fixed.
Each `schedule_circuit` call builds the configured layered schedule (`reuse`)
and its one-atom-AOD counterpart (`individual`, the same run with
`max_dimension` `[1, 1]`, built unless the AOD is already `[1, 1]`). Both are
validated and timed with the same model and the cheaper is returned, so a run
is never slower than moving one atom at a time. The layered planner moves
the operand needed again soonest (mover look-ahead) so loads can be retained.
The greedy side ordering was removed; a config that still sets `_side_ordering`
is rejected with an error. `T_reuse` controls load retention, including
identical-load continuation at zero. There is no group-ordering configuration.
Planning or replay errors propagate instead of silently dropping a candidate.

Once those load/reuse decisions are made, `dynamics.py` jointly reroutes each
continuous load that spans multiple layers. The tour is a symmetric TSP graph
(`_layered_tour_graph`) of `(layer, displacement)` cities plus a load terminal
(the current AOD offset) and an unload terminal (home). Terminals act as the
layers before the first and after the last. Only same-layer and consecutive-layer
edges exist; every layer crossing, including load and unload, pays a penalty
larger than a complete feasible tour, and the unload-to-load closing edge is free.
Any tour that splits or revisits a layer pays at least two extra penalties, so an
exact solver always visits layers in order. The default solver is `_solve_tour`
(Concorde, else `fast_tsp`); `tour_solver` accepts any symmetric solver, and
every returned tour is validated (Hamiltonian, existing edges, closing edge,
layer order) before use. Ordinary single-layer groups use the same graph with
one layer. Edges are priced at each pair of stops' cheapest parity-routed
interaction sides (best load/unload transfer included), which keeps one node
per stop and a symmetric matrix. Solvers need integer weights, so weights are
scaled to integers and absent edges get a large finite weight.
With `reorder_cz` or `reorder_cz_by_displacement` enabled, consecutive CZ-only
layers inside one retained load, with no 1Q context or barrier between them,
commute and form a single graph layer, so their stops can be visited in any
order. Replay validation then accepts per-wire order up to permutations of
these unconditional CZ runs. Duplicate targets in different layers stay distinct; duplicates
within a layer stay consecutive. If every solver fails, a returned
Hamiltonian tour that only broke layer order is repaired by sorting its cities
into layer order (keeping the tour's order within each layer, both directions
tried) with a warning; with no such tour, alignment ordering runs separately
within each layer with a warning. Either stand-in is still compared against
alignment (and the exact tour) like the normal relaxed tour.
The side DP then plans the complete retained-load route with one final return.
Load membership, retention decisions, gate dependencies,
and intervening single-qubit contexts are preserved and the result is replayed.
This improves routes for existing loads; it does not search for better load
partitions or prove globally minimum physical duration.

- `reorder_cz`: optional boolean, default `false`. After initial placement
  (including FastSA), the CLI swaps commuting, unconditional CZ gates to put
  equal signed home-to-partner displacements on the same explicit 2Q layer.
  Ancilla preference (`ancilla_registers`) determines the proxy mover when enabled;
  otherwise the first operand is used. Shared-operand gates and repeated
  occurrences remain in separate layers. Non-CZ wire dependencies, classical
  destinations, and barrier boundaries (on the barrier's qubits) are preserved.
- `reorder_cz_by_displacement`: compatibility option that enables the same
  placement-aware pass independently. Setting it to `false` does not disable
  displacement packing when `reorder_cz=true`. Both options require QASM
  input with full circuit context, not `step_order`.

The pass uses edge coloring within each displacement bucket to produce disjoint
matchings, then packs whole matchings largest-first into compatible layers.
Ties retain first-appearance order. It does not optimize a travel tour,
interaction sides, or AOD loads. Explicit
layer boundaries preserve shared-pulse opportunities, possibly increasing
logical depth. Parallelism is subject to subsequent AOD grouping constraints;
neither globally minimum pulse count nor minimum duration is guaranteed.

Only the reordered layers are scheduled, once, by the ordinary `dynamics.py`
grouping and shuttling code. `alignment_conc`, AOD span (`max_dimension`), `T_reuse`,
continue to control physical execution.
Disabling CZ reordering does not disable the group-level TSP tour or side DP.

`benchmarks/configs/bb144/bb144_n_init_qldpc/bb144_n_init_qldpc.json` enables `reorder_cz`. Run it with:

```bash
PYTHONPATH=src python -m naive_n_dag benchmarks/configs/bb144/bb144_n_init_qldpc/bb144_n_init_qldpc.json bb_n144_k12_d12_cz_syndrome.qasm benchmarks/output/bb144 --output-name bb144_qldpc_edge_colored
```

For the Python loader, pass `reorder_cz=True` to
`load_qasm_to_two_qubit_dag_with_single_qubit_context`. The default two-item return
is unchanged. `return_reference_nodes=True` additionally returns the full
prepared circuit for strict schedule replay. Preprocessing checks
operation occurrences and non-CZ wire boundaries against the original circuit;
motion validation then checks exact wire order against this prepared reference.
The CLI performs placement-aware reordering only after placement, using
`naive_n_dag.dag_helper.reorder_cz_layers_by_displacement`. It returns explicit
layers, rebuilt single-qubit context buckets, and the checked reordered
reference; these must be kept together. There is no separate block-routing
candidate or custom shuttler. The loader's optional geometry-free edge coloring
remains available to Python callers but is not the CLI's placement-aware pass.

Example configs:

- `benchmarks/configs/qft/qft_06_n_fastsa.json`
- `benchmarks/configs/bb144/bb144_n_fastsa_k1000.json`

## Tutorials

For usage walkthroughs:

- `Examples/schedule_tutorial.ipynb`: end-to-end scheduler usage (`naive_dag` and `naive_n_dag`), including `step_order` input.
- `Examples/qasm_to_circuit_and_dag_examples.ipynb`: QASM -> Qiskit circuit -> DAG examples, and the loader's transpile and ancilla options.
- `Examples/SA_Run_Example.ipynb`: running the FastSA placer on its own.

Everything the notebooks write goes to `Examples/OUTPUT/`.

## Benchmark Scripts

- `benchmarks/qldpc_notebook.ipynb` runs `naive_n_dag` on every QASM in `inputs/qasm_files/qLDPC/` with its FastSA and deterministic configs (`benchmarks/configs/qLDPC/`) and saves the execution and compile times to `benchmarks/output/qLDPC/qldpc_timings.csv`.
- `benchmarks/powermove_script.py` is used to generate runtime data for running `[[144,12,12]]` with Enola and PowerMove.
- This script requires cloning the PowerMove repository before use.

## AI Usage Note

The author wrote the base code: the `naive_dag` and `naive_n_dag` schedulers,
their lattice and movement model, and the schedule output format. AI
assistance (Claude Code) was then used, at the author's direction, to:

- add features and changes on top of that base, for example the deterministic
  placer, the `F_time` placement cost and FastSA tuning, TSP-based
  displacement tours, the optional Qiskit transpile step, `ancilla_registers`,
  multi-register QASM support, and the PowerMove/Enola benchmark scripts
- write tests for those changes
- write and revise the documentation: this README, `Rules.md`, the scheduler
  docs, docstrings and comments
- create and update the tutorial notebooks in `Examples/`
- review and clean up the code
