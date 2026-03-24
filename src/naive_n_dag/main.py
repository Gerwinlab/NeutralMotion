from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

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
    for key in ("qasm_dir", "qasm_base_dir"):
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
        "parallel",
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
    if fill_strategy not in {"random", "heuristic_fill"}:
        raise ValueError("fill_strategy must be either 'random' or 'heuristic_fill'.")

    if config["jerk"] != "constant":
        raise ValueError(
            f'Unsupported jerk profile "{config["jerk"]}". Only "constant" is allowed at this stage.'
        )


def main(
    config: Mapping[str, Any],
    qasm_file: PathLike,
    *,
    schedule_output_dir: PathLike | None = None,
    config_path: pathlib.Path | None = None,
    quiet: bool = False,
) -> int:
    """Validate naive-n inputs and print resolved run context.

    This is a scaffold entry point for the new method. It currently validates
    arguments/config and resolves paths, but does not yet run a scheduling pipeline.
    """
    if config_path is not None:
        config = resolve_config_paths(config, config_path.parent)

    _validate_required_config(config)
    qasm_path = _resolve_qasm_file(config, qasm_file)
    if not qasm_path.exists():
        raise FileNotFoundError(f"QASM file not found: {qasm_path}")

    for key in [
        "transfer_SLM_AOD",
        "max_acceleration",
        "max_velocity",
        "rydberg_radius",
        "average_single_gate_time",
        "average_two_gate_time",
        "t_switch",
    ]:
        parse_quantity(config[key], key)
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
        print(f"qasm_file={qasm_path}")
        print(f"max_dimension={config.get('max_dimension')}")
        print(f"fill_strategy={config.get('fill_strategy')}")
        if schedule_output_dir is not None:
            print(f"schedule_output_dir={pathlib.Path(schedule_output_dir).expanduser()}")

    return 0
