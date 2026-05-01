from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any, Mapping

from naive_n_dag.grid import Fastsa_Fill, generate_grid, initial_layout_fill, naive_fill
from naive_n_dag.dag_helper import (
    dag_from_txt_auto,
    load_qasm_to_two_qubit_dag_with_single_qubit_context,
)
from naive_n_dag.dynamics import best_path_for_layer
from naive_n_dag.scheduling import (
    ScheduleEvent,
    count_emitted_timesteps,
    single_qubit_layer_time,
    write_timed_schedule,
)
import pint

PathLike = str | pathlib.Path
ureg = pint.UnitRegistry()


def parse_quantity(value: Any, key: str):
    """Parse a quantity with units and reject dimensionless values."""
    try:
        q = ureg(value)
    except Exception as exc:  # pragma: no cover - passthrough for user config issues
        raise ValueError(f"{key} must include units (e.g. '60 microseconds'). Got: {value}") from exc

    if q.dimensionless:
        raise ValueError(f"{key} is missing units. Example: '60 microseconds'")
    return q


def load_config(path: PathLike) -> dict[str, Any]:
    """Load and validate the top-level JSON config object from disk."""
    config_path = pathlib.Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be an object, got {type(config).__name__}")
    return config


def _resolve_path(value: Any, base_dir: pathlib.Path) -> Any:
    """Resolve relative path-like strings against ``base_dir``."""
    if isinstance(value, str):
        candidate = pathlib.Path(value)
        if not candidate.is_absolute():
            return str((base_dir / candidate).resolve())
    return value


def resolve_config_paths(config: Mapping[str, Any], base_dir: pathlib.Path) -> dict[str, Any]:
    """Return a copy of ``config`` with qasm directory fields resolved to absolute paths."""
    resolved: dict[str, Any] = dict(config)
    for key in ("qasm_dir", "qasm_base_dir", "initial_layout", "step_order"):
        if key in resolved:
            resolved[key] = _resolve_path(resolved[key], base_dir)
    return resolved


def _resolve_qasm_file(config: Mapping[str, Any], qasm_file: PathLike) -> pathlib.Path:
    """Resolve the selected QASM file using explicit path or configured base directory."""
    qasm_path = pathlib.Path(qasm_file)
    if qasm_path.exists():
        return qasm_path.resolve()

    base_dir_value = config.get("qasm_base_dir") or config.get("qasm_dir")
    if base_dir_value is None:
        raise ValueError("Config must include qasm_base_dir or qasm_dir relative to config file.")
    return (pathlib.Path(base_dir_value) / qasm_path).resolve()


def _validate_required_config(config: Mapping[str, Any]) -> None:
    """Validate required scheduler config keys and core value constraints."""
    required_keys = [
        "dimensions",
        "num_NA",
        "transfer_SLM_AOD",
        "max_acceleration",
        "max_velocity",
        "rydberg_radius",
        "average_single_gate_time",
        "average_two_gate_time",
        "t_switch",
        "jerk",
        "movement_shape",
        "max_dimension",
        "fill_strategy",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    dims = config["dimensions"]
    if not isinstance(dims, list) or len(dims) != 2 or not all(isinstance(v, int) and v > 0 for v in dims):
        raise ValueError("dimensions must be a 2-item positive integer list: [rows, cols].")

    max_dim = config["max_dimension"]
    if not isinstance(max_dim, list) or len(max_dim) != 2 or not all(isinstance(v, int) and v > 0 for v in max_dim):
        raise ValueError("max_dimension must be a 2-item positive integer list: [x, y].")

    fill_strategy = config["fill_strategy"]
    if fill_strategy not in {"random", "heuristic_fill", "fastsa", "Given"}:
        raise ValueError("fill_strategy must be either 'random', 'heuristic_fill', or 'fastsa'.")

    if config["jerk"] != "constant":
        raise ValueError(
            f'Unsupported jerk profile "{config["jerk"]}". Only "constant" is allowed at this stage.'
        )

    if "alignment_conc" in config:
        alignment_conc = config["alignment_conc"]
        if not isinstance(alignment_conc, (int, float)):
            raise ValueError("alignment_conc must be a number in [-1, 1].")
        if alignment_conc < -1 or alignment_conc > 1:
            raise ValueError("alignment_conc must be in [-1, 1].")


def main(
    config: Mapping[str, Any],
    qasm_file: PathLike,
    *,
    schedule_output_dir: PathLike | None = None,
    config_path: pathlib.Path | None = None,
    quiet: bool = False,
    seed: int | None = None,
    output_name: str | None = None,
    log: bool = False,
) -> int:
    """Validate naive-n inputs and print resolved run context.

    This is a scaffold entry point for the new method. It currently validates
    arguments/config and resolves paths, but does not yet run a scheduling pipeline.
    """
    if config_path is not None:
        config = resolve_config_paths(config, config_path.parent)

    _validate_required_config(config)
    config = dict(config)
    config.setdefault("alignment_conc", 0.0)
    step_order_file = config.get("step_order")
    qasm_path: pathlib.Path | None = None
    source_path: pathlib.Path
    if step_order_file is None:
        qasm_path = _resolve_qasm_file(config, qasm_file)
        if not qasm_path.exists():
            raise FileNotFoundError(f"QASM file not found: {qasm_path}")
        source_path = qasm_path
    else:
        source_path = pathlib.Path(str(step_order_file)).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"step_order file not found: {source_path}")
        warnings.warn(
            "step_order is set; the provided qasm_file argument will be ignored.",
            stacklevel=2,
        )
        warnings.warn(
            "step_order input only contains two-qubit gates; single-qubit context is not loaded.",
            stacklevel=2,
        )

    for key in [
        "transfer_SLM_AOD",
        "max_acceleration",
        "max_velocity",
        "rydberg_radius",
        "average_single_gate_time",
        "average_two_gate_time",
        "t_switch",
    ]:
        config[key] = parse_quantity(config[key], key)
    config["transfer_SLM_AOD"] = config["transfer_SLM_AOD"].to("seconds")
    config["max_acceleration"] = config["max_acceleration"].to("meter/second^2")
    config["max_velocity"] = config["max_velocity"].to("meter/second")
    config["rydberg_radius"] = config["rydberg_radius"].to("meter")
    config["average_single_gate_time"] = config["average_single_gate_time"].to("seconds")
    config["average_two_gate_time"] = config["average_two_gate_time"].to("seconds")
    config["t_switch"] = config["t_switch"].to("seconds")

    if not quiet:
        qasm_dir = config.get("qasm_dir") or config.get("qasm_base_dir")
        print("naive_n_dag config loaded")
        print(f"dimensions={config.get('dimensions')} num_NA={config.get('num_NA')} qasm_dir={qasm_dir}")
        if qasm_path is not None:
            print(f"qasm_file={qasm_path}")
        else:
            print(f"step_order={source_path}")
            print("qasm_file=<ignored>")
        print(f"max_dimension={config.get('max_dimension')}")
        print(f"fill_strategy={config.get('fill_strategy')}")
        if schedule_output_dir is not None:
            print(f"schedule_output_dir={pathlib.Path(schedule_output_dir).expanduser()}")
    
    dims = config["dimensions"]
    num_na = config["num_NA"]
    grid = generate_grid(dims, config["rydberg_radius"])
    if step_order_file is None:
        if qasm_path is None:
            raise RuntimeError("Internal error: qasm_path was not resolved.")
        two_qubit_dag, single_layers = load_qasm_to_two_qubit_dag_with_single_qubit_context(qasm_path)
    else:
        two_qubit_dag, single_layers = dag_from_txt_auto(source_path)
    two_qubit_layers = list(two_qubit_dag.layers())
    qubits = None
    fastsa_result: Any | None = None
    fill_seed = seed if seed is not None else 0
    initial_layout = config.get("initial_layout")
    if config["fill_strategy"] == "Given":
        layout_path = pathlib.Path(str(initial_layout)).expanduser()
        if not layout_path.exists():
            raise FileNotFoundError(f"initial_layout file not found: {layout_path}")
        with layout_path.open("r", encoding="utf-8") as f:
            coords_array = json.load(f)
        if not isinstance(coords_array, list):
            raise ValueError("initial_layout JSON must be a list of [row, col] coordinates.")
        qubits = initial_layout_fill(grid, num_na, coords_array)
    elif config["fill_strategy"] == "heuristic_fill" and num_na > 1:
        raise ValueError("heuristic_fill strategy is only supported for num_NA=1 at this stage.")
    elif config["fill_strategy"] == "random":
        qubits = naive_fill(grid, num_na, fill_seed, True)
    elif config["fill_strategy"] == "fastsa":
        fastsa_cfg = config.get("Fastsa_fill", {})
        if fastsa_cfg is None:
            fastsa_cfg = {}
        if not isinstance(fastsa_cfg, Mapping):
            raise ValueError("Fastsa_fill must be a JSON object when provided.")
        fill_output = Fastsa_Fill(
            grid,
            num_na,
            two_qubit_dag,
            config,
            seed=fill_seed,
            stage1_iterations=int(fastsa_cfg.get("stage1_iterations", 200)),
            stage2_iterations=int(fastsa_cfg.get("stage2_iterations", 100)),
            initial_accept_prob=float(fastsa_cfg.get("initial_accept_prob", 0.99)),
            c=float(fastsa_cfg.get("c", 100.0)),
            stage3_temperature_threshold=float(fastsa_cfg.get("stage3_temperature_threshold", 1e-3)),
            stage3_max_iterations=int(fastsa_cfg.get("stage3_max_iterations", 1000)),
            stage3_section_size=int(fastsa_cfg.get("stage3_section_size", 4)),
            return_result=log,
        )
        if log:
            qubits, fastsa_result = fill_output
        else:
            qubits = fill_output
    if qubits is None:
        raise RuntimeError("Qubit placement is required before scheduling two-qubit layers.")

    #------------
    #starting the scheduling pipeline
    #------------
    time = 0 * (config["max_velocity"] / config["max_acceleration"]).units
    T = 0
    event_log: list[ScheduleEvent] = []
    current_positions: dict[int, tuple[int, int]] = {q.id: q.grid_position() for q in qubits}
    Previous_Positions: list[tuple[int, int]] = []
    Previous_Ids: list[int] = []

    for layer_idx, twoq_layer in enumerate(two_qubit_layers):
        single_step_lines, single_time = single_qubit_layer_time(
            single_layers[layer_idx],
            config["average_single_gate_time"],
            config["t_switch"],
        )
        for line in single_step_lines:
            event_log.append(("gate", line))
            T += 1
        time += single_time

        twoq_nodes = twoq_layer["graph"].op_nodes()
        if twoq_nodes:
            layer_steps, layer_time, Previous_Positions, Previous_Ids = best_path_for_layer(
                twoq_nodes,
                qubits,
                config,
                event_log,
                Previous_Ids=Previous_Ids,
                Previous_Positions=Previous_Positions,
                current_positions=current_positions,
            )
            T += layer_steps
            time += layer_time

    trailing_step_lines, trailing_time = single_qubit_layer_time(
        single_layers[-1],
        config["average_single_gate_time"],
        config["t_switch"],
    )
    for line in trailing_step_lines:
        event_log.append(("gate", line))
        T += 1
    time += trailing_time
    T = count_emitted_timesteps(event_log)

    final_time_us = time.to("microseconds")

    if not quiet:
        print(f"two_qubit_layers={len(two_qubit_layers)}")
        print(f"scheduled_timesteps={T}")
        print(f"estimated_time={final_time_us}")

    output_stem = config_path.stem if config_path is not None else source_path.stem
    if output_name is None:
        output_filename = f"{output_stem}.schedule.txt"
    else:
        candidate = pathlib.Path(output_name).name
        if candidate.endswith(".schedule.txt") or candidate.endswith(".txt"):
            output_filename = candidate
        else:
            output_filename = f"{candidate}.schedule.txt"

    if schedule_output_dir is None:
        warnings.warn(
            "No schedule output directory provided. Writing schedule in source directory.",
            stacklevel=2,
        )
        if config_path is not None:
            schedule_output = config_path.with_name(output_filename)
        else:
            schedule_output = source_path.with_name(output_filename)
    else:
        out_dir = pathlib.Path(schedule_output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = pathlib.Path.cwd() / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        schedule_output = out_dir / output_filename

    write_timed_schedule(
        schedule_output,
        solver="naive_n_dag",
        qasm_filename=source_path.name,
        final_time=str(final_time_us),
        lattice_spacing=str(config["rydberg_radius"].to("micrometers")),
        fill_seed=fill_seed,
        events=event_log,
        initial_qubits=qubits,
    )

    if not quiet:
        print(f"schedule_file={schedule_output}")

    if log and config["fill_strategy"] == "fastsa" and fastsa_result is not None:
        log_filename_base = output_filename
        if log_filename_base.endswith(".schedule.txt"):
            log_filename = f"{log_filename_base[:-13]}.fastsa_log.csv"
        elif log_filename_base.endswith(".txt"):
            log_filename = f"{log_filename_base[:-4]}.fastsa_log.csv"
        else:
            log_filename = f"{pathlib.Path(log_filename_base).stem}.fastsa_log.csv"
        log_output = schedule_output.with_name(log_filename)
        with log_output.open("w", encoding="utf-8") as f:
            f.write("step,best_cost,temperature\n")
            for step, (best_cost, temperature) in enumerate(
                zip(fastsa_result.best_cost_history, fastsa_result.temperature_history)
            ):
                f.write(f"{step},{best_cost},{temperature}\n")
        if not quiet:
            print(f"fastsa_log_file={log_output}")

    return 0
