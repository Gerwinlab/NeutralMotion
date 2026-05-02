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
python -m naive_n_dag inputs/algorithms/qft_06_n_fastsa.json inputs/qasm_files/qft_6.qasm outputs --seed 0 --output-name qft_06_n_fastsa_example --log
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

## Config Notes

Common required timing keys in JSON configs:

- `average_single_gate_time`
- `average_two_gate_time`
- `t_switch`
- `transfer_SLM_AOD`
- `max_acceleration`
- `max_velocity`
- `rydberg_radius`

Scheduling behavior flags:

- `parallel`: set to `true` to allow grouped/parallel movement scheduling in `naive_n_dag` (recommended for most benchmark-style runs).

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
