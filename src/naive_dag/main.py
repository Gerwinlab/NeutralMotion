from __future__ import annotations

import json
import pint
import pathlib
from typing import Any, Mapping
from .dag_helper import load_qasm_to_gate_dag
from .grid import generate_grid
from .grid import naive_fill
from .dynamics import best_path_for_gate, find_next_two_qubit_gate
from .scheduling import write_timed_schedule

#TODO: Add backend for qiskit such that users can write a another json file that specifies duration of gates.

PathLike = str | pathlib.Path
ureg = pint.UnitRegistry()

def parse_quantity(value, key):
    try:
        q = ureg(value)
    except Exception:
        raise ValueError(f"{key} must include units (e.g. '60 microseconds'). Got: {value}")

    # Check if units were actually provided
    if q.dimensionless:
        raise ValueError(f"{key} is missing units. Example: '60 microseconds'")

    return q

def load_config(path: PathLike) -> dict[str, Any]:
    config_path = pathlib.Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be an object, got {type(config).__name__}")
    return config


def _resolve_path(value: Any, base_dir: pathlib.Path) -> Any:
    if isinstance(value, str):
        candidate = pathlib.Path(value)
        if not candidate.is_absolute():
            return str((base_dir / candidate).resolve())
    return value


def resolve_config_paths(config: Mapping[str, Any], base_dir: pathlib.Path) -> dict[str, Any]:
    resolved: dict[str, Any] = dict(config)
    for key in ("qasm_dir", "qasm_base_dir"):
        if key in resolved:
            resolved[key] = _resolve_path(resolved[key], base_dir)
    return resolved


def _resolve_qasm_file(config: Mapping[str, Any], qasm_file: PathLike) -> pathlib.Path:
    qasm_path = pathlib.Path(qasm_file)

    # If user passed an existing path, use it directly
    if qasm_path.exists():
        return qasm_path.resolve()

    base_dir_value = config.get("qasm_base_dir") or config.get("qasm_dir")
    if base_dir_value is None:
        raise ValueError("Config must include qasm_base_dir or qasm_dir relative to config file.")

    base_dir = pathlib.Path(base_dir_value)

    qasm_path = base_dir / qasm_path
    return qasm_path.resolve()


def main(
    config: Mapping[str, Any],
    qasm_file: PathLike,
    *,
    config_path: pathlib.Path | None = None,
    quiet: bool = False,
) -> int:
    if config_path is not None:
        config = resolve_config_paths(config, config_path.parent)

    qasm_path = _resolve_qasm_file(config, qasm_file)

    dims = config.get("dimensions")
    num_na = config.get("num_NA")

    if not quiet:
        qasm_dir = config.get("qasm_dir") or config.get("qasm_base_dir")
        print("naive Dag config loaded")
        print(f"dimensions={dims} num_NA={num_na} qasm_dir={qasm_dir}")
        print(f"qasm_file={qasm_path}")
    if config["jerk"] != "constant":
            raise ValueError(
                f'Unsupported jerk profile "{config["jerk"]}". Only "constant" is allowed. You can implement none constant though.'
            )

    for key in ["transfer_SLM_AOD", "max_acceleration", "max_velocity", "rydberg_radius","Average_Gate_Time"]:
        config[key] = parse_quantity(config[key], key)
    
    config["transfer_SLM_AOD"] = config["transfer_SLM_AOD"].to("seconds")
    config["max_acceleration"] = config["max_acceleration"].to("meter/second^2")
    config["max_velocity"] = config["max_velocity"].to("meter/second")
    config["rydberg_radius"] = config["rydberg_radius"].to("meter")
    config["Average_Gate_Time"] = config["Average_Gate_Time"].to("seconds")
    qc = load_qasm_to_gate_dag(qasm_path)
    grid = generate_grid(dims,config["rydberg_radius"])
    qubits = naive_fill(grid,num_na,0,True)
    #------------
    #starting the scheduling pipeline
    #------------
    time = 0 * (config["max_velocity"] / config["max_acceleration"]).units
    T = 0
    event_log: list[tuple] = []
    i = 0
    while i < len(qc):
        if len(qc[i].qargs) != 2:
            i += 1
            continue
        t, T, gate_events = best_path_for_gate(qc, i, qubits, grid, config, T)
        time += t
        event_log.extend(gate_events)
        next_two_qubit = find_next_two_qubit_gate(qc, i)
        if next_two_qubit is None:
            break
        i = next_two_qubit

    schedule_output = qasm_path.with_suffix(".schedule.txt")
    write_timed_schedule(
        schedule_output,
        qasm_filename=qasm_path.name,
        final_time=str(time),
        lattice_spacing=str(config["rydberg_radius"].to("micrometers")),
        T=0,
        events=event_log,
    )

    print(T)
    print(time)
    print(f"schedule_file={schedule_output}")
    # TODO: wire into printing pipeline

    return 0
