# Phys765

Phys765 is a circuit-level scheduling and optimization package for neutral-atom quantum computing workflows. The project takes gate-level circuits (for example, circuits generated from pyLIQTR-style workflows), maps them onto a neutral-atom lattice model, and produces an execution schedule that reflects both gate application and atom transport constraints.

The core package is designed to answer a practical hardware question:
"Given a logical quantum circuit, where should atoms be placed, when should they move, and when should pulses be applied to complete the program efficiently?"

## Package Overview

At a high level, the naive_dag scheduler does the following:

- Loads a QASM circuit and converts it into a DAG-oriented gate sequence.
- Builds a 2D neutral-atom lattice from config dimensions and Rydberg spacing.
- Places logical qubits on valid even-even trap sites.
- Plans movement paths for two-qubit interactions (including load/move/unload style actions).
- Groups single-qubit gates into timestep layers so disjoint qubits can run together while same-qubit pulses remain sequential.
- Computes total runtime from motion dynamics and gate timing parameters.
- Writes a readable schedule with explicit timesteps (`T=...`) and actions.

The resulting schedule is intended to be easy to inspect and debug, with entries such as:

- `initialize q[...] -> (...)` at `T=0`
- `load` / `move` / `unload` motion actions
- gate lines (single- and two-qubit operations)

## What The Output Represents

The emitted schedule is a hardware-aware execution trace:

- Spatially aware: tracks lattice coordinates for atom transport.
- Temporally aware: each printed action is assigned a timestep.
- Timing aware: final runtime includes movement time, transfer costs, gate durations, and pulse-switch overhead.

This makes the package useful both for algorithm-level experimentation and for early-stage architecture studies where movement and pulse scheduling materially affect performance.

## Run

From the project root:

```bash
python -m src.naive_dag inputs/algorithms/qft_06.json qft_6.qasm outputs
```

This writes:

- `outputs/qft_6.schedule.txt`

## Config Notes

Current timing keys expected in JSON configs:

- `average_single_gate_time`
- `average_two_gate_time`
- `t_switch`
- `transfer_SLM_AOD`
- `max_acceleration`
- `max_velocity`
- `rydberg_radius`

Example config: `inputs/algorithms/qft_06.json`
