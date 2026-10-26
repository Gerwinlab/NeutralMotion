"""PowerMove AOD loads and unloads for the analysis-notebook benchmarks.

Runs PowerMove (via powermove_script.run_powermove with per_qubit=True) on
every benchmark with the storage zone off and on, and writes one CSV row per
run: load/unload steps, total atom loads/unloads and the most loads on one
qubit. Each movement stage loads and unloads every atom it moves once, so the
unload columns equal the load columns.

Each run gets a fresh process: PowerMove's placer seeds its RNG once at
import, so a run inside a reused process would place differently from a
standalone powermove_script.py run.

Runs are not repeatable: get_cz_blocks returns each CZ block as a set, so the
gate order inside a block (which the stage scheduler, the placer and the
router all follow) changes from process to process. ``--repeats N`` runs every
benchmark N times and keeps the run with the smallest T_exec. Rows already in
the output for benchmarks not run this time are kept, so ``--benchmarks``
updates only those benchmarks.

Usage (from the repo root):

    python benchmarks/powermove_transfers.py --jobs 4 --repeats 3
    python benchmarks/powermove_transfers.py --jobs 6 --repeats 3 --benchmarks QFT-63
"""

import argparse
import concurrent.futures as cf
import contextlib
import csv
import io
import multiprocessing
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

BENCHMARKS = {
    # label: QASM path relative to the repo root
    "BV-14": "inputs/qasm_files/bv/bv_n14.qasm",
    "BV-50": "inputs/qasm_files/bv/bv_n50.qasm",
    "BV-70": "inputs/qasm_files/bv/bv_n70.qasm",
    "QFT-18": "inputs/qasm_files/qft/qft_n18_transpiled.qasm",
    "QFT-29": "inputs/qasm_files/qft/qft_n29_transpiled.qasm",
    "QFT-63": "inputs/qasm_files/qft/qft_n63_transpiled.qasm",
    "VQE-30": "inputs/qasm_files/vqe/vqe_n30.qasm",
    "VQE-50": "inputs/qasm_files/vqe/vqe_n50.qasm",
    "Adder-18": "inputs/qasm_files/general/adder_18.qasm",
    "QV-20": "inputs/qasm_files/general/qv_20.qasm",
    "Heisenberg-24": "inputs/qasm_files/general/heisenberg_24.qasm",
    "QAOA-24": "inputs/qasm_files/general/qaoa_24.qasm",
    "Random-24": "inputs/qasm_files/general/random_24.qasm",
    "[[144,12,12]]": "inputs/qasm_files/qLDPC/bb_n144_k12_d12_cz_syndrome.qasm",
}

FIELDS = [
    "Benchmark", "qasm", "qubits", "storage", "gate_stages",
    "load_steps", "unload_steps", "atom_loads", "atom_unloads",
    "max_loads_one_qubit", "max_unloads_one_qubit", "max_load_qubits",
    "mean_loads_per_qubit", "Texec (µs)", "Tcomp (s)",
]


def run_one(label: str, storage: bool) -> dict:
    """One PowerMove run; return its CSV row."""
    from powermove_script import run_powermove

    qasm = BENCHMARKS[label]
    # PowerMove's stage scheduler prints even outside mvqc; keep the progress lines readable.
    with contextlib.redirect_stdout(io.StringIO()):
        r = run_powermove(os.path.join(REPO_ROOT, qasm), 1, 1, True, storage, per_qubit=True)
    atom_loads = sum(r["per_qubit_loads"])
    # The per-qubit counts must add up to the transfers PowerMove charges fidelity for.
    if atom_loads != r["atoms_loaded"]:
        raise RuntimeError(f"{label}: per-qubit loads {atom_loads} != atoms_loaded {r['atoms_loaded']}")
    return {
        "Benchmark": label,
        "qasm": qasm,
        "qubits": r["qubits"],
        "storage": "on" if storage else "off",
        "gate_stages": r["gate_stages"],
        "load_steps": r["load_steps"],
        "unload_steps": r["unload_steps"],
        "atom_loads": atom_loads,
        "atom_unloads": atom_loads,
        "max_loads_one_qubit": r["max_loads_one_qubit"],
        "max_unloads_one_qubit": r["max_loads_one_qubit"],
        "max_load_qubits": " ".join(map(str, r["max_load_qubits"])),
        "mean_loads_per_qubit": round(r["mean_loads_per_qubit"], 3),
        "Texec (µs)": round(r["total_exec_us"], 2),
        "Tcomp (s)": round(r["t_comp_s"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PowerMove AOD loads/unloads per benchmark, storage off and on")
    parser.add_argument("--jobs", type=int, default=1, help="Runs in parallel (default: 1)")
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "benchmarks", "output", "powermove_aod_transfers.csv"),
        help="CSV to write (default: benchmarks/output/powermove_aod_transfers.csv)",
    )
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), help="Subset of benchmarks (default: all)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per benchmark and storage setting; the one with the smallest T_exec is kept (default: 1)")
    args = parser.parse_args()

    runs = [(label, storage) for label in (args.benchmarks or BENCHMARKS) for storage in (False, True)] * args.repeats
    context = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(args.jobs, mp_context=context, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_one, label, storage): (label, storage) for label, storage in runs}
        best, failed = {}, []
        for future in cf.as_completed(futures):
            label, storage = futures[future]
            try:
                row = future.result()
            except Exception as error:  # a PowerMove failure on one run should not lose the others
                failed.append((label, storage))
                print(f"{label} storage={'on' if storage else 'off'}: FAILED ({type(error).__name__}: {error})", flush=True)
                continue
            if (label, storage) not in best or row["Texec (µs)"] < best[(label, storage)]["Texec (µs)"]:
                best[(label, storage)] = row
            print(f"{row['Benchmark']} storage={row['storage']}: Texec={row['Texec (µs)']} load_steps={row['load_steps']} "
                  f"atom_loads={row['atom_loads']} max_loads_one_qubit={row['max_loads_one_qubit']}", flush=True)
    rows = list(best.values())
    # Keep the rows of benchmarks this run did not cover.
    if os.path.exists(args.output):
        ran = {label for label, _ in runs}
        with open(args.output, newline="", encoding="utf-8") as fh:
            rows += [row for row in csv.DictReader(fh) if row["Benchmark"] not in ran and row["Benchmark"] in BENCHMARKS]

    rows.sort(key=lambda row: (list(BENCHMARKS).index(row["Benchmark"]), row["storage"] == "on"))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    if failed:
        sys.exit(f"{len(failed)} run(s) failed: " + ", ".join(f"{label} storage={'on' if s else 'off'}" for label, s in failed))


if __name__ == "__main__":
    main()
