#!/usr/bin/env python3
"""Prepare scheduler JSONs and run naive_n_dag on the qLDPC QASM collection.

QASMs: inputs/qasm_files/qLDPC/. Configs: benchmarks/configs/qLDPC/, two per
QASM: ``<stem>.json`` (FastSA placement) and ``<stem>_deterministic.json``
(the deterministic placer, otherwise identical). Outputs in
benchmarks/output/qLDPC/: ``<stem>.<strategy>.naive_n_dag.schedule.txt`` and,
for FastSA, ``<stem>.fastsa.naive_n_dag.fastsa_log.csv``. Nothing else is
written; progress, results and warnings are printed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOLVER_SRC = PROJECT / "src"
QASM_DIR = PROJECT / "inputs/qasm_files/qLDPC"
CONFIG_DIR = HERE / "configs/qLDPC"
TEMPLATE = HERE / "configs/vqe/vqe_n50.json"
PLACEMENTS = {"fastsa": ("fastsa", "random"), "deterministic": ("deterministic",)}


def save_json(path, data):
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def grid_dimensions(qubits):
    rows = math.isqrt(qubits)
    if rows * rows < qubits:
        rows += 1
    return [rows, (qubits + rows - 1) // rows]


def config_path(qasm, placement):
    suffix = "_deterministic" if placement == "deterministic" else ""
    return CONFIG_DIR / f"{qasm.stem}{suffix}.json"


def deterministic_twin(config):
    """The FastSA config with the deterministic placer (default budget) instead."""
    from naive_n_dag.deterministic_placement import DEFAULT_BUDGET

    twin = {}
    for key, value in config.items():
        if key == "fill_strategy":
            twin[key] = "deterministic"
        elif key == "Fastsa_fill":
            twin["Deterministic_fill"] = {"budget": DEFAULT_BUDGET}
        else:
            twin[key] = value
    twin.setdefault("Deterministic_fill", {"budget": DEFAULT_BUDGET})
    return twin


def prepare_config(qasm, circuit, placement, *, overwrite=False):
    """Create the QASM's config for ``placement`` if missing, then validate it.

    The FastSA config comes from TEMPLATE (benchmark physics); the
    deterministic one is a copy of the FastSA config. Existing configs are
    kept unless ``overwrite``, so hand edits survive reruns.
    """
    path = config_path(qasm, placement)
    if overwrite or not path.exists():
        if placement == "fastsa":
            config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
            dims = grid_dimensions(circuit.num_qubits)
            config.update(dimensions=dims, max_dimension=dims, num_NA=circuit.num_qubits,
                          qasm_base_dir=os.path.relpath(QASM_DIR, CONFIG_DIR), fill_strategy="fastsa")
            config["Fastsa_fill"]["stage3_section_size"] = dims[0]
        else:
            config = deterministic_twin(json.loads(prepare_config(qasm, circuit, "fastsa").read_text()))
        save_json(path, config)
    config = json.loads(path.read_text(encoding="utf-8"))
    # Validate before spending time in placement.
    from naive_n_dag.main import _coerce_unit_config, _validate_required_config
    _validate_required_config(config)
    _coerce_unit_config(dict(config))
    if config["num_NA"] != circuit.num_qubits:
        raise ValueError(f"{path.name}: num_NA must include all {circuit.num_qubits} qubits")
    if math.prod(config["dimensions"]) < circuit.num_qubits:
        raise ValueError(f"{path.name}: not enough lattice sites")
    if config.get("step_order"):
        raise ValueError("step_order would discard the QASM syndrome circuit")
    if config["fill_strategy"] not in PLACEMENTS[placement]:
        raise ValueError(f"{path.name}: fill_strategy must be one of {PLACEMENTS[placement]}")
    base = Path(config.get("qasm_base_dir", config.get("qasm_dir", ".")))
    if not base.is_absolute():
        base = path.parent / base
    if (base / qasm.name).resolve() != qasm.resolve():
        raise ValueError(f"{path.name}: qasm_base_dir does not resolve to its matching QASM")
    return path


def validate_saved_schedule(path, circuit):
    """Check serialized operation counts, measurement destinations, and each wire's order."""
    expected = defaultdict(list)
    counts = Counter()
    for inst in circuit.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        qubits = tuple(circuit.find_bit(q).index for q in inst.qubits)
        classical = None
        if inst.clbits:
            locations = circuit.find_bit(inst.clbits[0]).registers
            if len(locations) != 1:
                raise ValueError("Measurement bit must belong to exactly one classical register")
            register, local_index = locations[0]
            classical = (register.name, local_index)
        signature = (name, qubits, classical)
        counts[name] += 1
        for q in qubits:
            expected[q].append(signature)
    actual = defaultdict(list)
    actual_counts = Counter()
    content = path.read_text(encoding="utf-8")
    pattern = (r"\b(h|cz|reset|measure)\s+"
               r"(q\[\d+\](?:,\s*q\[\d+\])*)"
               r"(?:\s*->\s*([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\])?;")
    for match in re.finditer(pattern, content):
        name, operands, register, destination = match.groups()
        qubits = tuple(map(int, re.findall(r"q\[(\d+)\]", operands)))
        classical = (register, int(destination)) if destination is not None else None
        signature = (name, qubits, classical)
        actual_counts[name] += 1
        for q in qubits:
            actual[q].append(signature)
    if actual_counts != counts or dict(actual) != dict(expected):
        raise ValueError("Saved schedule does not preserve all operations, wire order, and measurement bits")
    return dict(actual_counts)


def run_one(qasm, circuit, config, output, args):
    """Run one QASM/config pair; True on success. Only the schedule (and FastSA log) are saved."""
    strategy = json.loads(config.read_text())["fill_strategy"]
    name = qasm.stem + f".{strategy}.naive_n_dag"
    schedule = output / f"{name}.schedule.txt"
    fastsa_log = output / f"{name}.fastsa_log.csv"
    label = f"{qasm.stem} [{strategy}]"
    if schedule.exists() and not args.overwrite:
        print(f"SKIP {label}: {schedule.name} exists (use --overwrite)", flush=True)
        return True
    command = [sys.executable, "-u", "-m", "naive_n_dag", str(config),
               str(qasm), str(output), "--seed", str(args.seed),
               "--output-name", name]
    if strategy == "fastsa":
        command.append("--log")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOLVER_SRC) + os.pathsep + environment.get("PYTHONPATH", "")
    print(f"RUN  {label} ({circuit.num_qubits} atoms)", flush=True)
    start = time.monotonic()
    try:
        process = subprocess.run(command, cwd=PROJECT, env=environment, capture_output=True,
                                 text=True, timeout=args.timeout or None)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT {label}: exceeded {args.timeout:g} s", flush=True)
        return False
    elapsed = time.monotonic() - start
    if process.returncode:
        tail = (process.stdout + process.stderr).strip().splitlines()[-15:]
        print(f"FAILED {label}: naive_n_dag exited with code {process.returncode}", flush=True)
        print("\n".join("    " + line for line in tail), flush=True)
        return False
    final = re.search(r"final_time:\s*(\S+)", schedule.read_text(encoding="utf-8"))
    print(f"OK   {label}: {float(final.group(1)):.1f} us in {elapsed:.1f} s", flush=True)
    # Diagnostic only: report problems without discarding a usable schedule
    # (for example, when CZ reordering makes the saved wire order differ
    # from the source QASM).
    try:
        validate_saved_schedule(schedule, circuit)
    except Exception as error:
        print(f"     warning: {error}", flush=True)
    if strategy == "fastsa":
        with fastsa_log.open(encoding="utf-8") as log:
            if next(log, "").strip() != "step,best_cost,temperature" or not next(log, "").strip():
                print("     warning: missing or empty FastSA iteration log", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pattern", default="*.qasm", help="QASM glob within inputs/qasm_files/qLDPC")
    parser.add_argument("--placement", choices=["fastsa", "deterministic", "both"], default="both",
                        help="Which placer(s) to run per QASM (default: both)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output/qLDPC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=0, help="Per-run seconds; 0 means unlimited")
    parser.add_argument("--prepare-only", action="store_true", help="Generate/validate JSONs without solving")
    parser.add_argument("--overwrite-configs", action="store_true", help="Regenerate JSONs from the template")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing schedules")
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("--timeout must be nonnegative")
    sys.path.insert(0, str(SOLVER_SRC))
    try:
        from qiskit import qasm2
        import naive_n_dag.main  # noqa: F401
    except ImportError as error:
        parser.error(f"{error}. Run using {PROJECT.parent / 'Phys765_Test/bin/python'}")
    qasms = sorted(path for path in QASM_DIR.glob(args.pattern)
                   if path.is_file() and path.suffix == ".qasm")
    if not qasms or any(path.parent != QASM_DIR or path.suffix != ".qasm" for path in qasms):
        parser.error("--pattern must select QASM files directly in inputs/qasm_files/qLDPC")
    placements = ["fastsa", "deterministic"] if args.placement == "both" else [args.placement]
    items = []
    for qasm in qasms:
        circuit = qasm2.load(str(qasm))
        for placement in placements:
            items.append((qasm, circuit, prepare_config(qasm, circuit, placement,
                                                        overwrite=args.overwrite_configs)))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Prepared/validated {len(items)} scheduler JSONs", flush=True)
    if args.prepare_only:
        return 0
    failures = 0
    for qasm, circuit, config in items:
        try:
            failures += not run_one(qasm, circuit, config, output, args)
        except Exception as error:
            failures += 1
            print(f"FAILED {qasm.stem}: {error}", flush=True)
    print(f"Finished: {len(items) - failures} successful, {failures} failed. Outputs: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
