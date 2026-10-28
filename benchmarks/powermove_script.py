"""Baseline: run the PowerMove and Enola compilers on a QASM circuit and report their times.

Both tools are imported from a clone of the PowerMove repository
(github.com/Scarlett0815/PowerMove, branch ``powermove-cz-throttling``) next to
this repo (``../PowerMove``): ``mvqc`` (PowerMove, one AOD), ``mvqc_multi_aod``
and ``enola``. The circuit is transpiled to {u1, u2, u3, cz, id} and split into
CZ blocks; each tool's transfer and movement times come from its own model, and
gate time is the number of CZ stages run times T_RYDBERG_US. PowerMove's CZ
throttling can split a stage, so its CZ stages are counted as they run. Results
are printed; with ``--csv`` and ``--benchmark`` they also replace (or add) that
benchmark's PowerMove and Enola rows in a timings CSV such as
``benchmarks/output/scheduler_batch_timings.csv`` (solver "PowerMove" for
storage off, "PowerMove (storage on)" with ``--storage``). PowerMove runs are not
repeatable (its CZ blocks are sets), so ``--repeats N`` runs it N times and
keeps the smallest T_exec. With ``--per-qubit`` PowerMove also reports how many
times each qubit is loaded into (and unloaded from) the AOD.

Usage (from the repo root):

    python benchmarks/powermove_script.py \\
        --qasm inputs/qasm_files/qLDPC/bb_n144_k12_d12_cz_syndrome.qasm

    python benchmarks/powermove_script.py \\
        --qasm inputs/qasm_files/qft/qft_n63_transpiled.qasm --repeats 3 \\
        --csv benchmarks/output/scheduler_batch_timings.csv --benchmark QFT-63
"""

import argparse
import collections
import contextlib
import csv
import io
import math
import os
import random
import sys
import time

import networkx
from qiskit import QuantumCircuit, QuantumRegister, transpile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
POWERMOVE_ROOT = os.path.join(PROJECT_ROOT, "PowerMove")#change this to the name of your clone
if POWERMOVE_ROOT not in sys.path:
    sys.path.insert(0, POWERMOVE_ROOT)

from Construct_Circuit import get_cz_blocks
import mvqc as powermove_module
import mvqc_multi_aod as powermove_multi_aod_module
from enola import enola, storage_gate_scheduling as enola_stage_scheduling
from router.codegen import T_ACTIVATE

powermove_stage_scheduling = powermove_module.storage_gate_scheduling


T_RYDBERG_US = 0.36  # time charged per CZ gate stage
# Both artifacts charge this fidelity per atom transfer, so the number of atom
# transfers is log(cir_fidelity_atom_transfer) / log(FIDELITY_ATOM_TRANSFER).
FIDELITY_ATOM_TRANSFER = 0.999


def _atom_transfers(fidelity_atom_transfer: float) -> int:
    """Number of atom transfers implied by an artifact's total transfer fidelity."""
    return round(math.log(fidelity_atom_transfer) / math.log(FIDELITY_ATOM_TRANSFER))


@contextlib.contextmanager
def _track_stages(gate_stages: list, loads: collections.Counter):
    """While active, record every stage PowerMove's one-AOD ``mvqc`` runs.

    ``mvqc`` takes each stage's CZs off a deque (CZ throttling may put a
    reduced stage back and retry it), builds a cycle graph whose nodes are the
    stage's moves (qubit, target), and calls ``update`` once per stage it runs.
    Wraps those three names in the module (does not edit it). Per stage run:
    one CZ stage if it has gates, and one load (and unload) per moved qubit.
    Counting moved qubits as loads is right only because PowerMove loads and
    unloads every atom it moves in every stage (no atom stays in the AOD
    between stages); it does not carry over to a scheduler that holds atoms.
    """
    module = powermove_module
    last = {"gates": [], "moves": []}

    class RecordingDeque(collections.deque):
        def popleft(self):
            last["gates"] = super().popleft()
            return last["gates"]

    class RecordingNetworkx:
        def __getattr__(self, name):
            return getattr(networkx, name)

        def simple_cycles(self, graph):
            last["moves"] = list(graph.nodes)
            return networkx.simple_cycles(graph)

    original_update = module.update

    def update(*args, **kwargs):
        gate_stages[0] += bool(last["gates"])
        for qubit, _ in last["moves"]:
            loads[qubit] += 1
        last["moves"] = []
        return original_update(*args, **kwargs)

    saved = module.deque, module.nx, module.update
    module.deque, module.nx, module.update = RecordingDeque, RecordingNetworkx(), update
    try:
        yield
    finally:
        module.deque, module.nx, module.update = saved


def load_cz_blocks(qasm_path: str):
    """Transpile the QASM to the artifacts' basis; return (num_qubits, CZ blocks)."""
    circ = QuantumCircuit.from_qasm_file(qasm_path)
    # get_cz_blocks reads register-local q._index, so give it one register (global order).
    flat = QuantumCircuit(QuantumRegister(circ.num_qubits, "q"), *circ.cregs)
    for inst in circ.data:
        flat.append(inst.operation, [flat.qubits[circ.find_bit(q).index] for q in inst.qubits], inst.clbits)
    circ = flat
    test_circuit = transpile(
        circ,
        basis_gates=["u1", "u2", "u3", "cz", "id"],
        optimization_level=2,
    )
    # get_cz_blocks prints every block.
    with contextlib.redirect_stdout(io.StringIO()):
        cz_blocks = get_cz_blocks(test_circuit)
    return test_circuit.num_qubits, cz_blocks


def _require_unit_distance(d: int) -> None:
    """The current PowerMove repository has no distance scaling (the artifact's ``d``)."""
    if d != 1:
        raise ValueError(f"d={d} is not supported: the current PowerMove repository has no distance scaling")


def run_powermove(
    qasm_path: str,
    d: int,
    num_aod: int,
    quiet: bool,
    storage_flag: bool = False,
    per_qubit: bool = False,
) -> dict:
    """Compile with PowerMove (``mvqc``) and return its stage counts and times in us.

    With ``per_qubit`` the result also holds each qubit's AOD load count (one
    AOD only).
    """
    _require_unit_distance(d)
    if per_qubit and num_aod != 1:
        raise ValueError("per-qubit load counts need num_aod=1")
    t0 = time.time()
    n, cz_blocks = load_cz_blocks(qasm_path)
    row = math.ceil(math.sqrt(n))

    gate_stages, loads = [0], collections.Counter()
    if num_aod == 1:
        tracking = _track_stages(gate_stages, loads)

        def run():
            return powermove_module.mvqc(cz_blocks, row, n, storage_flag)
    else:
        # mvqc_multi_aod has no CZ throttling: its stages are the scheduled ones.
        gate_stages[0] = sum(len(powermove_multi_aod_module.storage_gate_scheduling(g, storage_flag)) for g in cz_blocks)
        tracking = contextlib.nullcontext()

        def run():
            return powermove_multi_aod_module.mvqc(cz_blocks, row, n, storage_flag, num_aod)
    with tracking:
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                result = run()
        else:
            result = run()
    gate_stages = gate_stages[0]
    transfer_us, move_us, fidelity_transfer, movement_stages = result[0], result[1], result[6], result[8]

    gate_time_us = gate_stages * T_RYDBERG_US
    transport_us = transfer_us + move_us

    extra = {}
    if per_qubit:
        # Each stage that moves a qubit loads and unloads it once, so the
        # unload counts equal these.
        per_qubit_loads = [loads.get(q, 0) for q in range(n)]
        max_loads = max(per_qubit_loads)
        extra = {
            "per_qubit_loads": per_qubit_loads,
            "max_loads_one_qubit": max_loads,
            "max_load_qubits": [q for q, c in enumerate(per_qubit_loads) if c == max_loads],
            "mean_loads_per_qubit": sum(per_qubit_loads) / n,
        }

    return {
        "qasm": qasm_path,
        "qubits": n,
        "row": row,
        "storage_flag": storage_flag,
        "d": d,
        "num_aod": num_aod,
        "gate_stages": gate_stages,
        # mvqc: every movement stage picks atoms up and drops them off, and
        # adds 2 * MUS_PER_FRM = 16 us of transfer time per stage; each moved
        # atom is charged one transfer fidelity per stage (per load/unload pair).
        "load_steps": movement_stages,
        "unload_steps": movement_stages,
        "atoms_loaded": _atom_transfers(fidelity_transfer),
        "atoms_unloaded": _atom_transfers(fidelity_transfer),
        "transfer_us": transfer_us,
        "move_us": move_us,
        "transport_us": transport_us,
        "gate_time_us": gate_time_us,
        "total_exec_us": transport_us + gate_time_us,
        "t_comp_s": time.time() - t0,
        **extra,
    }


def run_enola(qasm_path: str, d: int, quiet: bool) -> dict:
    """Compile with Enola and return its stage counts and times in us."""
    _require_unit_distance(d)
    t0 = time.time()
    n, cz_blocks = load_cz_blocks(qasm_path)
    row = math.ceil(math.sqrt(n))

    gate_stages = 0
    for gates in cz_blocks:
        gate_stages += len(enola_stage_scheduling(gates, False))

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result = enola(cz_blocks, row, n)
    else:
        result = enola(cz_blocks, row, n)
    transfer_us, move_us, fidelity_transfer = result[0], result[1], result[6]

    gate_time_us = gate_stages * T_RYDBERG_US
    transport_us = transfer_us + move_us

    return {
        "qasm": qasm_path,
        "qubits": n,
        "row": row,
        "d": d,
        "gate_stages": gate_stages,
        # Enola emits Activate (load) and Deactivate (unload) instructions of
        # T_ACTIVATE us each and charges the fidelity per atom in each, so
        # these totals count loads and unloads together.
        "load_unload_steps": round(transfer_us / T_ACTIVATE),
        "atom_loads_plus_unloads": _atom_transfers(fidelity_transfer),
        "transfer_us": transfer_us,
        "move_us": move_us,
        "transport_us": transport_us,
        "gate_time_us": gate_time_us,
        "total_exec_us": transport_us + gate_time_us,
        "t_comp_s": time.time() - t0,
    }


def print_result(name: str, result: dict) -> None:
    print(f"{name}:")
    print(f"  qasm={result['qasm']}")
    print(f"  qubits={result['qubits']} row={result['row']}")
    if "storage_flag" in result:
        print(f"  storage={result['storage_flag']}")
    print(f"  gate_stages={result['gate_stages']}")
    for key in ("load_steps", "unload_steps", "atoms_loaded", "atoms_unloaded",
                "load_unload_steps", "atom_loads_plus_unloads"):
        if key in result:
            print(f"  {key}={result[key]}")
    print(f"  transfer_us={result['transfer_us']:.6f}")
    print(f"  move_us={result['move_us']:.6f}")
    print(f"  transport_us={result['transport_us']:.6f}")
    print(f"  gate_time_us={result['gate_time_us']:.6f}")
    print(f"  total_exec_us={result['total_exec_us']:.6f}")
    print(f"  t_comp_s={result['t_comp_s']:.6f}")
    if "per_qubit_loads" in result:
        print(f"  max_loads_one_qubit={result['max_loads_one_qubit']} (= max unloads)")
        print(f"  max_load_qubits={result['max_load_qubits']}")
        print(f"  mean_loads_per_qubit={result['mean_loads_per_qubit']:.3f}")
        print(f"  per_qubit_loads={result['per_qubit_loads']}")
    print()


TIMINGS_FIELDS = ["solver", "config_json", "qasm_arg", "Tcomp (s)", "output_file", "Texec (µs)", "Benchmark"]


def update_timings_csv(path: str, benchmark: str, results: dict) -> None:
    """Replace (or add) one row per solver for ``benchmark`` in a timings CSV.

    ``results`` maps a solver name ("PowerMove", "Enola") to its run_* result.
    Other rows are kept as they are.
    """
    rows, fields = [], TIMINGS_FIELDS
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or TIMINGS_FIELDS
            rows = [r for r in reader if not (r["Benchmark"] == benchmark and r["solver"] in results)]
    for solver, result in results.items():
        rows.append({
            "solver": solver,
            "config_json": "benchmarks/powermove_script.py",
            "qasm_arg": os.path.basename(result["qasm"]),
            "Tcomp (s)": round(result["t_comp_s"], 2),
            "output_file": "",
            "Texec (µs)": round(result["total_exec_us"], 2),
            "Benchmark": benchmark,
        })
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"updated {', '.join(results)} rows for {benchmark} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PowerMove then Enola on a QASM circuit (default: the [[144,12,12]] BB syndrome circuit)"
    )
    parser.add_argument(
        "--qasm",
        default=os.path.join(REPO_ROOT, "inputs", "qasm_files", "qLDPC", "bb_n144_k12_d12_cz_syndrome.qasm"),
        help="Path to QASM file (default: inputs/qasm_files/qLDPC/bb_n144_k12_d12_cz_syndrome.qasm)",
    )
    parser.add_argument("--d", type=int, default=1,
                        help="Distance scaling factor d; the current PowerMove repository supports only 1 (default: 1)")
    parser.add_argument("--num-aod", type=int, default=1, help="PowerMove number of AODs (default: 1)")
    parser.add_argument(
        "--storage",
        action="store_true",
        help="Enable the storage-zone model (default: disabled, matching the paper's non-storage column)",
    )
    parser.add_argument(
        "--paper-placement",
        action="store_true",
        help="Reproduce the RNG advance in PowerMove's original BV benchmark script",
    )
    parser.add_argument(
        "--per-qubit",
        action="store_true",
        help="Also report PowerMove's AOD load (= unload) count per qubit, its maximum and mean",
    )
    parser.add_argument(
        "--skip-enola",
        action="store_true",
        help="Run only PowerMove instead of also running the Enola comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show raw PowerMove/Enola logs",
    )
    parser.add_argument("--skip-powermove", action="store_true", help="Run only Enola")
    parser.add_argument("--repeats", type=int, default=1,
                        help="PowerMove runs; the one with the smallest T_exec is kept (default: 1)")
    parser.add_argument("--csv", help="Timings CSV whose PowerMove/Enola rows for --benchmark are replaced or added")
    parser.add_argument("--benchmark", help="Benchmark label for --csv, e.g. QFT-63")
    args = parser.parse_args()

    qasm_path = os.path.abspath(args.qasm)
    if not os.path.exists(qasm_path):
        raise FileNotFoundError(f"QASM file not found: {qasm_path}")
    if args.csv and not args.benchmark:
        parser.error("--csv needs --benchmark")
    if args.csv and args.num_aod != 1:
        parser.error("--csv records one-AOD PowerMove runs")

    quiet = not args.verbose

    if args.paper_placement:
        # The artifact's BV benchmark makes this otherwise-unused draw before
        # simulated-annealing placement, changing the deterministic layout.
        random.choice(range(10))

    if args.skip_powermove and args.skip_enola:
        parser.error("nothing to run: both --skip-powermove and --skip-enola")
    results = {}
    if not args.skip_powermove:
        runs = [run_powermove(qasm_path, args.d, args.num_aod, quiet, args.storage, args.per_qubit)
                for _ in range(args.repeats)]
        if args.repeats > 1:
            print("PowerMove T_exec per run (us): " + ", ".join(f"{r['total_exec_us']:.1f}" for r in runs))
        pm = min(runs, key=lambda r: r["total_exec_us"])
        print_result("PowerMove", pm)
        # The timings CSV keeps storage-off runs as "PowerMove".
        results["PowerMove (storage on)" if args.storage else "PowerMove"] = pm
    if not args.skip_enola:
        en = run_enola(qasm_path, args.d, quiet)
        print_result("Enola", en)
        results["Enola"] = en
    if args.csv:
        update_timings_csv(args.csv, args.benchmark, results)


if __name__ == "__main__":
    main()
