#!/usr/bin/env python3
"""Prepare scheduler JSONs and run naive_n_dag on the qLDPC QASM collection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
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
TEMPLATE = PROJECT / "Scheduler_Test/INPUT/vqe_n50.json"


def save_json(path, data):
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_dimensions(qubits):
    rows = math.isqrt(qubits)
    if rows * rows < qubits:
        rows += 1
    return [rows, (qubits + rows - 1) // rows]


def prepare_config(qasm, circuit, *, overwrite=False):
    """Use benchmark physics and FastSA placement by default."""
    config_path = qasm.with_suffix(".json")
    if overwrite or not config_path.exists():
        config = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        dims = grid_dimensions(circuit.num_qubits)
        config.update(dimensions=dims, max_dimension=dims, num_NA=circuit.num_qubits,
                      qasm_base_dir=".", fill_strategy="fastsa")
        config["Fastsa_fill"]["stage3_section_size"] = dims[0]
        save_json(config_path, config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Validate before spending time in FastSA. Keep user edits on ordinary reruns.
    from naive_n_dag.main import _coerce_unit_config, _validate_required_config
    _validate_required_config(config)
    _coerce_unit_config(dict(config))
    if config["num_NA"] != circuit.num_qubits:
        raise ValueError(f"{config_path.name}: num_NA must include all {circuit.num_qubits} qubits")
    if math.prod(config["dimensions"]) < circuit.num_qubits:
        raise ValueError(f"{config_path.name}: not enough lattice sites")
    if config.get("step_order"):
        raise ValueError("step_order would discard the QASM syndrome circuit")
    if config["fill_strategy"] not in {"fastsa", "random"}:
        raise ValueError("This collection runner supports random and fastsa placement")
    base = Path(config.get("qasm_base_dir", config.get("qasm_dir", ".")))
    if not base.is_absolute():
        base = config_path.parent / base
    if (base / qasm.name).resolve() != qasm.resolve():
        raise ValueError(f"{config_path.name}: qasm_base_dir does not resolve to its matching QASM")
    return config_path


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


def run_one(qasm, circuit, config, output, args, solver_hash):
    strategy = json.loads(config.read_text())["fill_strategy"]
    name = qasm.stem + f".{strategy}.naive_n_dag"
    paths = {"schedule": output / f"{name}.schedule.txt",
             "console_log": output / f"{name}.log",
             "result": output / f"{name}.result.json"}
    if strategy == "fastsa":
        paths["fastsa_log"] = output / f"{name}.fastsa_log.csv"
    artifact_keys = set(paths) - {"result"}
    fingerprint = {"qasm_sha256": sha256(qasm), "config_sha256": sha256(config),
                   "solver_sha256": solver_hash, "runner_sha256": sha256(Path(__file__)),
                   "seed": args.seed, "python": sys.executable}
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        old = json.loads(paths["result"].read_text()) if paths["result"].exists() else {}
        if (old.get("status") == "success" and old.get("fingerprint") == fingerprint
                and all(paths[key].exists() and sha256(paths[key]) == digest
                        for key, digest in old.get("artifact_sha256", {}).items())
                and set(old.get("artifact_sha256", {})) == artifact_keys):
            print(f"SKIP {qasm.name}: matching validated outputs exist", flush=True)
            return old
        raise FileExistsError(f"Existing outputs for {qasm.stem}; use --overwrite or --output-dir")

    command = [sys.executable, "-u", "-m", "naive_n_dag", str(config),
               str(qasm), str(output), "--seed", str(args.seed),
               "--output-name", name]
    if strategy == "fastsa":
        command.append("--log")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOLVER_SRC) + os.pathsep + environment.get("PYTHONPATH", "")
    result = {"qasm": qasm.name, "config": config.name, "fingerprint": fingerprint,
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": command,
              "status": "running", "fill_strategy": strategy,
              "outputs": {k: str(v) for k, v in paths.items()}}
    save_json(paths["result"], result)
    print(f"RUN  {qasm.name} ({circuit.num_qubits} atoms)", flush=True)
    start = time.monotonic()
    with paths["console_log"].open("w", encoding="utf-8") as log:
        log.write(f"Source QASM: {qasm}\nConfig: {config}\nSeed: {args.seed}\n\n")
        log.flush()
        try:
            process = subprocess.run(command, cwd=PROJECT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=args.timeout or None)
            result["returncode"] = process.returncode
            if process.returncode:
                raise RuntimeError(f"naive_n_dag exited with code {process.returncode}")
            result["status"] = "success"
            # A zero-exit solver process is a successful batch run.  Artifact
            # validation is diagnostic: report problems without discarding an
            # otherwise usable schedule (for example, when CZ reordering makes
            # the saved wire order differ from the source QASM).
            validation_warnings = []
            try:
                result["gate_counts"] = validate_saved_schedule(paths["schedule"], circuit)
            except Exception as error:
                validation_warnings.append(str(error))
            if strategy == "fastsa":
                try:
                    with paths["fastsa_log"].open(encoding="utf-8") as fastsa:
                        if (next(fastsa).strip() != "step,best_cost,temperature"
                                or not next(fastsa, "").strip()):
                            raise ValueError("Missing or empty FastSA iteration log")
                except Exception as error:
                    validation_warnings.append(str(error))
            if validation_warnings:
                result["validation_warnings"] = validation_warnings
        except subprocess.TimeoutExpired:
            result.update(status="timeout", error=f"Exceeded {args.timeout} seconds")
        except Exception as error:
            result.update(status="failed", error=str(error))
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        log.write("\nBatch result: " + result["status"] + "\n")
        if "error" in result:
            log.write(result["error"] + "\n")
        for warning in result.get("validation_warnings", []):
            log.write("Validation warning: " + warning + "\n")
    if result["status"] == "success":
        artifact_sha256 = {}
        for key in sorted(artifact_keys):
            try:
                artifact_sha256[key] = sha256(paths[key])
            except Exception as error:
                result.setdefault("validation_warnings", []).append(
                    f"Could not fingerprint {key} artifact: {error}")
        result["artifact_sha256"] = artifact_sha256
    save_json(paths["result"], result)
    print(f"{result['status'].upper()} {qasm.name}: {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="*.qasm", help="QASM glob within this script's directory")
    parser.add_argument("--output-dir", type=Path, default=HERE / "OUTPUT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=0, help="Per-circuit seconds; 0 means unlimited")
    parser.add_argument("--prepare-only", action="store_true", help="Generate/validate JSONs without solving")
    parser.add_argument("--overwrite-configs", action="store_true", help="Regenerate JSONs from the template")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing solver outputs")
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("--timeout must be nonnegative")
    sys.path.insert(0, str(SOLVER_SRC))
    try:
        from qiskit import qasm2
        import naive_n_dag.main  # noqa: F401
    except ImportError as error:
        parser.error(f"{error}. Run using {PROJECT.parent / 'Phys765_Test/bin/python'}")
    qasms = sorted(path for path in HERE.glob(args.pattern)
                   if path.is_file() and path.suffix == ".qasm")
    if not qasms or any(path.parent != HERE or path.suffix != ".qasm" for path in qasms):
        parser.error("--pattern must select QASM files directly in qLDPC_Qasms")
    items = []
    for qasm in qasms:
        circuit = qasm2.load(str(qasm))
        config = prepare_config(qasm, circuit, overwrite=args.overwrite_configs)
        items.append((qasm, circuit, config))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Prepared/validated {len(items)} scheduler JSONs", flush=True)
    if args.prepare_only:
        return 0
    sources = sorted(SOLVER_SRC.glob("naive*_dag/*.py"))
    solver_hash = hashlib.sha256(b"".join(str(p.relative_to(SOLVER_SRC)).encode() + p.read_bytes()
                                         for p in sources)).hexdigest()
    summary = []
    for qasm, circuit, config in items:
        try:
            result = run_one(qasm, circuit, config, output, args, solver_hash)
        except Exception as error:
            result = {"qasm": qasm.name, "status": "failed", "error": str(error)}
            print(f"FAILED {qasm.name}: {error}", flush=True)
        summary.append(result)
        save_json(output / "batch_summary.json", summary)
    failures = sum(item["status"] != "success" for item in summary)
    print(f"Finished: {len(summary)-failures} successful, {failures} failed. Outputs: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
