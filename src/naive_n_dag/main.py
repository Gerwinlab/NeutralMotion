from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any, Mapping

from naive_n_dag.grid import Fastsa_Fill, generate_grid, initial_layout_fill, naive_fill
from naive_n_dag.deterministic_placement import deterministic_fill, validate_options
from naive_n_dag.placement_cost import COST_FUNCTION_REMOVED
from naive_n_dag.dag_helper import (
    dag_from_txt_auto,
    load_qasm_to_two_qubit_dag_with_single_qubit_context,
    reorder_cz_layers_by_displacement,
)
from naive_n_dag.dynamics import (
    _PARALLEL_REMOVED, _SIDE_ORDERING_REMOVED, _TSP_GADGET_REPLACED, RYDBERG_RADIUS,
    TOUR_ORDERINGS, schedule_circuit,
)
from naive_n_dag.scheduling import (
    count_emitted_timesteps,
    write_timed_schedule,
)
import pint
#TODO: Update the version, AI Usage for this version, and the readme to reflect the new changes. Make sure all examples work properly.
#TODO: Make a new function or runable command that reads movements and the unload/load to calculate to measure of fidelity and loss.
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


_RETIRED_KEYS = {
    "rydberg_radius": (
        "Config key 'rydberg_radius' was renamed to 'lattice_spacing' (the SLM trap pitch). "
        "The Rydberg radius is now the constant RYDBERG_RADIUS = 15 micrometers in "
        "naive_n_dag.dynamics."
    ),
    "_side_ordering": _SIDE_ORDERING_REMOVED,
    "parallel": _PARALLEL_REMOVED,
    "tsp_gadget": _TSP_GADGET_REPLACED,
    "cost_function": COST_FUNCTION_REMOVED,
}


def _reject_renamed_keys(config: Mapping[str, Any]) -> None:
    """Fail loudly on renamed or removed keys instead of silently aliasing them."""
    for key, message in _RETIRED_KEYS.items():
        if key in config:
            raise ValueError(message)


def _coerce_unit_config(config: dict[str, Any]) -> None:
    """Parse and normalize all unit-bearing config fields in place."""
    _reject_renamed_keys(config)
    config.setdefault("measurement_time", "0 milliseconds")
    for key in [
        "transfer_SLM_AOD",
        "max_acceleration",
        "max_velocity",
        "lattice_spacing",
        "average_single_gate_time",
        "average_two_gate_time",
        "measurement_time",
        "t_switch",
    ]:
        config[key] = parse_quantity(config[key], key)

    config["transfer_SLM_AOD"] = config["transfer_SLM_AOD"].to("seconds")
    config["max_acceleration"] = config["max_acceleration"].to("meter/second^2")
    config["max_velocity"] = config["max_velocity"].to("meter/second")
    config["lattice_spacing"] = config["lattice_spacing"].to("meter")
    config["average_single_gate_time"] = config["average_single_gate_time"].to("seconds")
    config["average_two_gate_time"] = config["average_two_gate_time"].to("seconds")
    config["measurement_time"] = config["measurement_time"].to("seconds")
    config["t_switch"] = config["t_switch"].to("seconds")


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
    _reject_renamed_keys(config)
    if not isinstance(config.get("reorder_cz", False), bool):
        raise ValueError("reorder_cz must be a boolean.")
    if not isinstance(config.get("reorder_cz_by_displacement", config.get("reorder_cz", False)), bool):
        raise ValueError("reorder_cz_by_displacement must be a boolean.")
    if not isinstance(config.get("prefer_moving_ancillas", False), bool):
        raise ValueError("prefer_moving_ancillas must be a boolean.")
    if config.get("reorder_cz", False) and config.get("step_order") is not None:
        raise ValueError("reorder_cz requires QASM input with full gate context, not step_order.")
    if config.get("reorder_cz_by_displacement", config.get("reorder_cz", False)) and config.get("step_order") is not None:
        raise ValueError("reorder_cz_by_displacement requires QASM input with full gate context, not step_order.")
    required_keys = [
        "dimensions",
        "num_NA",
        "transfer_SLM_AOD",
        "max_acceleration",
        "max_velocity",
        "lattice_spacing",
        "average_single_gate_time",
        "average_two_gate_time",
        "t_switch",
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
    if fill_strategy not in {"random", "heuristic_fill", "fastsa", "Given", "deterministic"}:
        raise ValueError(
            "fill_strategy must be one of 'random', 'heuristic_fill', 'fastsa', 'Given', "
            "or 'deterministic'."
        )
    deterministic_options = config.get("Deterministic_fill")
    if deterministic_options is not None:
        if not isinstance(deterministic_options, Mapping):
            raise ValueError("Deterministic_fill must be a JSON object when provided.")
        validate_options(deterministic_options)

    movement_shape = config["movement_shape"]
    if movement_shape not in {"trapezoidal", "Cubic_Spline"}:
        raise ValueError(
            "movement_shape must be either 'trapezoidal' or 'Cubic_Spline'."
        )

    if "alignment_conc" in config:
        alignment_conc = config["alignment_conc"]
        if not isinstance(alignment_conc, (int, float)):
            raise ValueError("alignment_conc must be a number in [-1, 1].")
        if alignment_conc < -1 or alignment_conc > 1:
            raise ValueError("alignment_conc must be in [-1, 1].")

    if "tsp_time_limit" in config:
        limit = config["tsp_time_limit"]
        if not isinstance(limit, (int, float)) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("tsp_time_limit must be a positive number of seconds.")

    if "concorde_time_limit" in config:
        limit = config["concorde_time_limit"]
        if not isinstance(limit, (int, float)) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("concorde_time_limit must be a positive number of seconds.")

    if "tour_ordering" in config and config["tour_ordering"] not in TOUR_ORDERINGS:
        raise ValueError(f"tour_ordering must be one of {', '.join(map(repr, TOUR_ORDERINGS))}.")

    if "T_reuse" in config:
        reuse_horizon = config["T_reuse"]
        if isinstance(reuse_horizon, str):
            if reuse_horizon != "Inf":
                raise ValueError('T_reuse must be a nonnegative integer or "Inf".')
        elif not isinstance(reuse_horizon, int) or isinstance(reuse_horizon, bool) or reuse_horizon < 0:
            raise ValueError('T_reuse must be a nonnegative integer or "Inf".')


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
    """Run the naive-n scheduling pipeline and write a textual schedule.

    Input can come from either:
    - a QASM file (standard path), or
    - a `step_order` text file that contains only two-qubit timestep data.
    """
    if config_path is not None:
        config = resolve_config_paths(config, config_path.parent)

    _validate_required_config(config)
    config = dict(config)
    config.setdefault("alignment_conc", 0.0)
    config.setdefault("T_reuse", 0)
    if config["T_reuse"] == "Inf":
        config["T_reuse"] = float("inf")
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

    _coerce_unit_config(config)

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
    grid = generate_grid(dims, config["lattice_spacing"])
    reference_nodes = None
    if step_order_file is None:
        if qasm_path is None:
            raise RuntimeError("Internal error: qasm_path was not resolved.")
        two_qubit_dag, single_layers, reference_nodes = load_qasm_to_two_qubit_dag_with_single_qubit_context(
            qasm_path, reorder_cz=False,
            return_reference_nodes=True,
        )
        if not quiet:
            print(f"reorder_cz={config.get('reorder_cz', False)}")
    else:
        two_qubit_dag, single_layers = dag_from_txt_auto(source_path)
    reduced_layers = list(two_qubit_dag.layers())
    two_qubit_layers = [
        [node for node in layer["graph"].op_nodes() if node.op.name != "barrier"]
        for layer in reduced_layers
    ]
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
            initial_accept_prob=float(fastsa_cfg.get("initial_accept_prob", 0.999)),
            c=float(fastsa_cfg.get("c", 100.0)),
            stage3_temperature_threshold=float(fastsa_cfg.get("stage3_temperature_threshold", 0.0)),
            stage3_max_iterations=int(fastsa_cfg.get("stage3_max_iterations", 1000)),
            stage3_section_size=int(fastsa_cfg.get("stage3_section_size", 4)),
            return_result=log,
            reference_nodes=reference_nodes,
        )
        if log:
            qubits, fastsa_result = fill_output
        else:
            qubits = fill_output
    elif config["fill_strategy"] == "deterministic":
        # Same objective and two-qubit DAG as fastsa; program-order reference
        # nodes (absent with step_order) give symmetry discovery its gate blocks.
        qubits, placement = deterministic_fill(
            grid, num_na, two_qubit_dag, config,
            reference_nodes=reference_nodes,
            options=config.get("Deterministic_fill"),
            return_result=True,
        )
        if not quiet:
            print(
                f"deterministic_placement route={placement.route} "
                f"objective={placement.objective:.6g} "
                f"evaluations={placement.evaluations}/{placement.budget} "
                f"descents={placement.descents} stop={placement.stop_reason}"
            )
    if qubits is None:
        raise RuntimeError("Qubit placement is required before scheduling two-qubit layers.")

    # CZ displacement packing is an unconditional preprocessing step when
    # enabled, not a candidate competing with unreordered physical schedules.
    if config.get("reorder_cz", False) or config.get("reorder_cz_by_displacement", False):
        two_qubit_layers, single_layers, reference_nodes = reorder_cz_layers_by_displacement(
            reference_nodes, {q.id: q.grid_position() for q in qubits}, config,
        )
        if not quiet:
            print(f"displacement_precompile_layers={sum(bool(layer) for layer in two_qubit_layers)}")

    # Group and route only after the placement-aware gate order is finalized.
    schedule = schedule_circuit(
        two_qubit_layers, single_layers, qubits, config, reference_nodes,
    )
    event_log = schedule.events
    time = schedule.duration
    emitted_timesteps = count_emitted_timesteps(event_log)
    if not quiet:
        print(f"selected_schedule={schedule.selected}")
        print(f"reused_groups={schedule.reused_groups}")
        if schedule.tsp_stats is not None:
            stats = schedule.tsp_stats
            print(
                f"tsp_solves={stats.solves} tsp_enumerated={stats.enumerated} "
                f"tsp_gadget={stats.gadget} tsp_exact_wins={stats.exact_wins} "
                f"tsp_alignment_wins={stats.alignment_wins} "
                f"tsp_exact_failures={stats.exact_failures} "
                f"tsp_seconds_saved={stats.seconds_saved * 1e6:.6g}us"
            )
        for name, duration in schedule.candidate_times.items():
            print(f"candidate_{name}_time={duration.to('microseconds')}")
        for name, reason in schedule.rejected_candidates.items():
            print(f"rejected_candidate_{name}={reason}")

    final_time_us = time.to("microseconds")

    if not quiet:
        print(f"two_qubit_layers={sum(bool(layer) for layer in two_qubit_layers)}")
        print(f"scheduled_timesteps={emitted_timesteps}")
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
        lattice_spacing=str(config["lattice_spacing"].to("micrometers")),
        rydberg_radius=f"{RYDBERG_RADIUS * 1e6:g} micrometer",
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
            # best_cost in the cost's own units; temperature on FastSA's
            # normalised scale (cost / mean random-layout cost).
            f.write("step,best_cost,temperature\n")
            for step, (best_cost, temperature) in enumerate(
                zip(fastsa_result.best_cost_history, fastsa_result.temperature_history)
            ):
                f.write(f"{step},{best_cost * fastsa_result.cost_scale},{temperature}\n")
        if not quiet:
            print(f"fastsa_log_file={log_output}")

    return 0
