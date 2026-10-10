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
  An atom unloaded and then reloaded by the next transfer step stays in the
  AOD instead: only atoms that change traps get `load`/`unload`, and the kept
  atom appears as a `move` in the same step (it passes over its own home site).

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

Once those load/reuse decisions are made, `dynamics.py` jointly reroutes each
continuous load that spans multiple layers. The tour is a symmetric TSP graph
(`_layered_tour_graph`) of `(layer, displacement)` cities plus a load terminal
(the current AOD offset) and an unload terminal (home). Terminals act as the
layers before the first and after the last. Only same-layer and consecutive-layer
edges exist; every layer crossing, including load and unload, pays a penalty
larger than a complete feasible tour, and the unload-to-load closing edge is free.
Any tour that splits or revisits a layer pays at least two extra penalties, so an
exact solver always visits layers in order. The default solver is Held-Karp
(`_held_karp_tour`); `tour_solver` accepts any symmetric solver, such as a
Concorde wrapper, and every returned tour is validated (Hamiltonian, existing
edges, closing edge, layer order) before use. Ordinary single-layer groups use
the same graph with one layer. Edges are priced between target centers and,
separately, at each pair of stops' cheapest parity-routed interaction sides
(best load/unload transfer included). Both pricings keep one node per stop and
a symmetric matrix; both tours are scored by the exact fixed-order side DP and
the cheaper order is kept (center order on ties, and always under greedy sides).
With `reorder_cz` or `reorder_cz_by_displacement` enabled, consecutive CZ-only
layers inside one retained load, with no 1Q context or barrier between them,
commute and form a single graph layer, so their stops can be visited in any
order. Replay validation then accepts per-wire order up to permutations of
these unconditional CZ runs. Duplicate targets in different layers stay distinct; duplicates
within a layer stay consecutive. The 24-city limit counts these labelled cities;
above it, alignment ordering runs separately within each layer with a warning.
The side DP then plans the complete retained-load route with one final return.
Greedy sides remain opt-in. Load membership, retention decisions, gate dependencies,
and intervening single-qubit contexts are preserved and the result is replayed.
This improves routes for existing loads; it does not search for better load
partitions or prove globally minimum physical duration.

- `reorder_cz`: optional boolean, default `false`. After initial placement
  (including FastSA), the CLI swaps commuting, unconditional CZ gates to put
  equal signed home-to-partner displacements on the same explicit 2Q layer.
  Reset-identified ancilla preference determines the proxy mover when enabled;
  otherwise the first operand is used. Shared-operand gates and repeated
  occurrences remain in separate layers. Non-CZ wire dependencies, classical
  destinations, and global barrier boundaries are preserved.
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
grouping and shuttling code. `parallel`, `alignment_conc`, AOD span, `T_reuse`,
and the configured side policy continue to control physical execution.
Disabling CZ reordering does not disable group-level Held-Karp or side DP.

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
The CLI performs placement-aware reordering only after placement, using
`naive_n_dag.dag_helper.reorder_cz_layers_by_displacement`. It returns explicit
layers, rebuilt single-qubit context buckets, and the checked reordered
reference; these must be kept together. There is no separate block-routing
candidate or custom shuttler. The loader's optional geometry-free edge coloring
remains available to Python callers but is not the CLI's placement-aware pass.

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
