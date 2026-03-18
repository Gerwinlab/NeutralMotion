from __future__ import annotations

import argparse
import json
import pathlib

from .main import load_config, main as naive_dag_main, resolve_config_paths

#TODO: Update the pyproject.toml and README.md
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="naive_dag",
        description="Run the naive scheduler with a JSON configuration file.",
    )
    parser.add_argument(
        "config",
        type=pathlib.Path,
        help="Path to a JSON configuration file.",
    )
    parser.add_argument(
        "qasm_file",
        help="QASM filename or path (resolved relative to qasm_base_dir in the JSON).",
    )
    parser.add_argument(
        "schedule_output_dir",
        nargs="?",
        default=None,
        help="Optional directory to write <qasm_stem>.schedule.txt. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument(
        "--dump-config",
        action="store_true",
        help="Print the resolved configuration and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-error output.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path: pathlib.Path = args.config
    config = load_config(config_path)
    qasm_file = args.qasm_file
    schedule_output_dir = args.schedule_output_dir

    if args.dump_config:
        resolved = resolve_config_paths(config, config_path.parent)
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return

    exit_code = naive_dag_main(
        config,
        qasm_file,
        schedule_output_dir=schedule_output_dir,
        config_path=config_path,
        quiet=args.quiet,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
