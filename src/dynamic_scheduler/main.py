from __future__ import annotations

import json
import pathlib
from typing import Any, Mapping

#TODO: check if number of qubits matches the number of qubits in the qasm file.
PathLike = str | pathlib.Path

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


def main(config: Mapping[str, Any], *, config_path: pathlib.Path | None = None, quiet: bool = False) -> int:
    if config_path is not None:
        config = resolve_config_paths(config, config_path.parent)

    if not quiet:
        dims = config.get("dimensions")
        num_na = config.get("num_NA")
        qasm_dir = config.get("qasm_dir") or config.get("qasm_base_dir")
        print("Dynamic scheduler config loaded")
        print(f"dimensions={dims} num_NA={num_na} qasm_dir={qasm_dir}")

    

    # TODO: wire into scheduling pipeline
    return 0
