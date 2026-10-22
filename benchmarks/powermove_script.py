import argparse
import contextlib
import io
import math
import os
import random
import sys
import time

from qiskit import QuantumCircuit, transpile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(REPO_ROOT, ".."))
POWERMOVE_ROOT = os.path.join(PROJECT_ROOT, "PowerMove_AE")
if POWERMOVE_ROOT not in sys.path:
    sys.path.insert(0, POWERMOVE_ROOT)

from Construct_Circuit import get_cz_blocks
from PowerMove import mvqc
from Enola import enola, storage_gate_scheduling as enola_stage_scheduling
from Stage_Scheduler import storage_gate_scheduling as powermove_stage_scheduling
from router.codegen import T_ACTIVATE


T_RYDBERG_US = 0.36
# Both artifacts charge this fidelity per atom transfer, so the number of atom
# transfers is log(cir_fidelity_atom_transfer) / log(FIDELITY_ATOM_TRANSFER).
FIDELITY_ATOM_TRANSFER = 0.999


def _atom_transfers(fidelity_atom_transfer: float) -> int:
    return round(math.log(fidelity_atom_transfer) / math.log(FIDELITY_ATOM_TRANSFER))


def load_cz_blocks(qasm_path: str):
    circ = QuantumCircuit.from_qasm_file(qasm_path)
    test_circuit = transpile(
        circ,
        basis_gates=["u1", "u2", "u3", "cz", "id"],
        optimization_level=2,
    )
    return test_circuit.num_qubits, get_cz_blocks(test_circuit)


def run_powermove(
    qasm_path: str,
    d: int,
    num_aod: int,
    quiet: bool,
    storage_flag: bool = False,
) -> dict:
    t0 = time.time()
    n, cz_blocks = load_cz_blocks(qasm_path)
    row = math.ceil(math.sqrt(n))
    list_gates = []
    for gates in cz_blocks:
        list_gates += powermove_stage_scheduling(gates, storage_flag)
    gate_stages = len(list_gates)

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result = mvqc(cz_blocks, row, n, storage_flag, d, num_aod)
    else:
        result = mvqc(cz_blocks, row, n, storage_flag, d, num_aod)
    transfer_us, move_us, fidelity_transfer, movement_stages = result[0], result[1], result[6], result[8]

    gate_time_us = gate_stages * T_RYDBERG_US
    transport_us = transfer_us + move_us

    return {
        "qasm": qasm_path,
        "qubits": n,
        "row": row,
        "storage_flag": storage_flag,
        "d": d,
        "num_aod": num_aod,
        "gate_stages": gate_stages,
        # Coll_Moves_Scheduler: every movement stage is one load (pickup) batch
        # and one unload (dropoff) batch, charged 2 x 8 us; each moved atom is
        # charged one transfer fidelity per stage, i.e. per load/unload pair.
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
    }


def run_enola(qasm_path: str, d: int, quiet: bool) -> dict:
    t0 = time.time()
    n, cz_blocks = load_cz_blocks(qasm_path)
    row = math.ceil(math.sqrt(n))

    gate_stages = 0
    for gates in cz_blocks:
        gate_stages += len(enola_stage_scheduling(gates, False))

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            result = enola(cz_blocks, row, n, d)
    else:
        result = enola(cz_blocks, row, n, d)
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
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PowerMove then Enola on the [[144,12,12]] BB syndrome circuit"
    )
    parser.add_argument(
        "--qasm",
        default=os.path.join(REPO_ROOT, "inputs", "qasm_files", "qLDPC", "bb_n144_k12_d12_cz_syndrome.qasm"),
        help="Path to QASM file (default: inputs/qasm_files/qLDPC/bb_n144_k12_d12_cz_syndrome.qasm)",
    )
    parser.add_argument("--d", type=int, default=1, help="Distance scaling factor d (default: 1)")
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
        "--skip-enola",
        action="store_true",
        help="Run only PowerMove instead of also running the Enola comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show raw PowerMove/Enola logs",
    )
    args = parser.parse_args()

    qasm_path = os.path.abspath(args.qasm)
    if not os.path.exists(qasm_path):
        raise FileNotFoundError(f"QASM file not found: {qasm_path}")

    quiet = not args.verbose

    if args.paper_placement:
        # The artifact's BV benchmark makes this otherwise-unused draw before
        # simulated-annealing placement, changing the deterministic layout.
        random.choice(range(10))

    pm = run_powermove(qasm_path, args.d, args.num_aod, quiet, args.storage)
    print_result("PowerMove", pm)
    if not args.skip_enola:
        en = run_enola(qasm_path, args.d, quiet)
        print_result("Enola", en)


if __name__ == "__main__":
    main()
