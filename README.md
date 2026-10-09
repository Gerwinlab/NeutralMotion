# Phys765

Phys765 is a neutral-atom circuit scheduling package. It loads QASM circuits, maps logical qubits to lattice sites, schedules gate/movement actions, and emits a timestep schedule text file.

## Install

From the repository root:

```bash
python3 -m pip install -e .
```

This installs the package in editable mode and exposes these CLI commands:

- `naive_dag`
- `naive_n_dag`

## Quick Run (`naive_n_dag`)

Example:

```bash
naive_n_dag inputs/algorithms/qft_06_n_fastsa.json inputs/qasm_files/qft_6.qasm outputs --seed 0 --output-name qft_06_n_fastsa_example --log
```

This writes:

- `outputs/qft_06_n_fastsa_example.schedule.txt`
- `outputs/qft_06_n_fastsa_example.fastsa_log.csv` (because `--log` is enabled with FastSA)

## What The Schedulers Do

At a high level, `naive_dag` / `naive_n_dag`:

- Load QASM and convert to DAG-style gate ordering.
- Build a neutral-atom lattice from config dimensions and Rydberg spacing.
- Place qubits on valid trap sites.
- Schedule movement + gate application over timesteps.
- Emit a readable schedule (`T=...`, `initialize`, `load/move/unload`, gate lines).

QASM barriers are treated as hard global scheduling boundaries. They are not
printed as physical actions, but no gate or measurement may be moved from one
side of a barrier to the other.

## Config Notes

Common required timing keys in JSON configs:

- `average_single_gate_time`
- `average_two_gate_time`
- `t_switch`
- `transfer_SLM_AOD`
- `max_acceleration`
- `max_velocity`
- `rydberg_radius`

`measurement_time` is optional and defaults to `0 milliseconds`. All
measurements emitted in the same timestep share one measurement-time charge,
plus one `t_switch`, regardless of how many qubits are measured in that step.

Scheduling behavior flags:

- `movement_shape`: selects the shuttle timing profile. `trapezoidal` uses the
  velocity-limited acceleration profile. `Cubic_Spline` uses
  `sqrt(6 * delta_x / acceleration) + sqrt(6 * delta_y / acceleration)`, with
  physical axis distances in meters.
- `prefer_moving_ancillas`: optional boolean, default `false`. When `true`, a
  two-qubit gate with exactly one reset-identified ancilla always moves that
  ancilla instead of its data-qubit partner. If neither or both operands are
  identified as ancillas, the normal mover heuristic is used.
- `parallel`: set to `true` to allow grouped/parallel movement scheduling in `naive_n_dag` (recommended for most benchmark-style runs).

`naive_n_dag` defaults to Held-Karp displacement ordering and fixed-order
side optimization for its ordinary legal 2Q groups, including CX and other
supported gates, with either value of the CZ reordering flags. Held-Karp uses
at most 24 unique targets (warn and fall back to alignment above that); the side
DP still applies after the fallback. Fresh groups optimize loading direction;
reused groups start at their actual AOD offset without reloading. Home return
is included in the cost estimate but deferred to allow reuse. Only disjoint
gates within a group are reordered; noncommuting wire dependencies remain fixed.
Each `schedule_circuit` call builds one configured schedule, without benchmarking
other policies. Set `_side_ordering` to `"greedy"` to use greedy interaction sides;
otherwise the side DP is used. `T_reuse` controls load retention, including
identical-load continuation at zero. There is no group-ordering configuration:
alignment ordering is used only when the exact displacement-tour limit is exceeded.
Planning or replay errors are reported instead of switching policies.

- `reorder_cz`: optional boolean, default `false`. In `naive_n_dag`, reorder
  unconditional CZs within commuting blocks before extracting the two-qubit DAG
  and single-qubit context. Other operations constrain block formation. Bipartite
  blocks use exact edge coloring by repeated perfect matchings; other graphs use
  greedy coloring. If the proposed ordering increases the extracted two-qubit
  layer count, the loader retains the original ordering and single-qubit
  context. Equal-depth proposals are accepted. This protects logical depth,
  but does not guarantee shorter atom movement. Requires
  QASM input, not `step_order`; conditional gates/control flow are unsupported.
- `reorder_cz_by_displacement`: optional boolean, defaults to the value of
  `reorder_cz`. After initial placement (including FastSA), a second pass packs
  commuting CZs with the same signed `partner_home - mover_home` displacement
  into explicit layers. It honors reset-identified ancilla mover preference,
  disjoint operands, non-CZ dependencies and global barriers. Equal-displacement
  matchings are kept together using a greedy packing heuristic; this is not a
  global pulse-count optimum and may increase logical depth. Both the original
  and packed layerings are fully scheduled and validated. A `phase_tour`
  candidate additionally orders unique displacements across each entire
  commuting CZ block, avoiding restrictive cross-layer target pairings. It
  holds a consistent moving bipartition in one load, including atoms needed
  later in the tour, and returns home at each block boundary. Held-Karp uses
  only displacement states (at most 24 unique targets; above that, warn and use
  alignment). For each traversal direction, a separate fixed-order dynamic
  program chooses four interaction-side states per visit, including loading
  and returning home. Its roughly `16 * visits` transitions are costed and
  executed using the unchanged parity router. This does not add side states
  to Held-Karp's exponential search or change its visit order. Both directions
  use the configured side policy (DP by default, greedy only when requested)
  and are replay-validated; the fastest complete route is retained.
  This optional wider search supplements the ordinary
  group-level Held-Karp and side DP; disabling preprocessing does not disable
  either group-level optimizer. The parity router and legality rules are unchanged.
  Non-bipartite blocks, incompatible mover preferences, or loads violating
  `parallel`, AOD span or `T_reuse` reject this optional candidate. Resets,
  readout and barriers remain intact. The fastest valid complete schedule is
  selected. Packed candidates have the `displacement_` prefix. Set this option
  to `false` to retain only the earlier layer-count-based preprocessing, or set
  it to `true` independently of `reorder_cz`. Requires full QASM context.

`inputs/algorithms/bb144_n_init_qldpc.json` enables `reorder_cz`. Run it with:

```bash
PYTHONPATH=src python -m naive_n_dag inputs/algorithms/bb144_n_init_qldpc.json qLDPC_Qasms/bb_n144_k12_d12_cz_syndrome.qasm outputs/bb144 --output-name bb144_qldpc_edge_colored
```

For the Python loader, pass `reorder_cz=True` to
`load_qasm_to_two_qubit_dag_with_single_qubit_context`. The default two-item return
is unchanged. `return_reference_nodes=True` additionally returns the full
prepared, SWAP-filtered circuit for strict schedule replay. Preprocessing checks
operation occurrences and non-CZ wire boundaries against the original circuit;
motion validation then checks exact wire order against this prepared reference.
The placement-aware pass is separate from this loader: the CLI invokes
`naive_n_dag.precompile.precompile_displacement_layers` after placing atoms and
before grouping/shuttling. It returns explicit layers, rebuilt single-qubit
context buckets and the checked reordered reference; these must be kept together.
`naive_n_dag.phase_schedule.schedule_phase_tours` uses the same dependency-safe
block extraction but routes across the old layer boundaries. Its reported
`two_qubit_layers` counts disjoint displacement visits, not the earlier packed
layers; increasing this count need not increase CZ pulses or movement time.

Example configs:

- `inputs/algorithms/qft_06_n_fastsa.json`
- `inputs/algorithms/bb144_n_fastsa_k1000.json`

## Tutorials

For usage walkthroughs:

- `schedule_tutorial.ipynb`: end-to-end scheduler usage (`naive_dag` and `naive_n_dag`).
- `qasm_to_circuit_and_dag_examples.ipynb`: QASM -> Qiskit circuit -> DAG examples.

## Benchmark Scripts

- `powermove_script.py` is used to generate runtime data for running `[[144,12,12]]` with Enola and PowerMove.
- This script requires cloning the PowerMove repository before use.

## AI Usage Note

AI assistance was used to:

- create and refine tutorial notebooks
- improve code readability
- improve inline documentation and docstrings
- limited usage in the writing of the actual scheduler
