import argparse
import contextlib
import io
import math
import os
import sys
import time

from qiskit import QuantumCircuit, transpile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
POWERMOVE_ROOT = os.path.join(PROJECT_ROOT, "PowerMove_AE")
if POWERMOVE_ROOT not in sys.path:
    sys.path.insert(0, POWERMOVE_ROOT)

from Construct_Circuit import get_cz_blocks
from PowerMove import mvqc
from Enola import enola, storage_gate_scheduling as enola_stage_scheduling
from Stage_Scheduler import storage_gate_scheduling as powermove_stage_scheduling


T_RYDBERG_US = 0.36


def load_cz_blocks(qasm_path: str):
    circ = QuantumCircuit.from_qasm_file(qasm_path)
    test_circuit = transpile(
        circ,
        basis_gates=["u1", "u2", "u3", "cz", "id"],
        optimization_level=2,
    )
    return test_circuit.num_qubits, get_cz_blocks(test_circuit)


def run_powermove(qasm_path: str, d: int, num_aod: int, quiet: bool) -> dict:
    t0 = time.time()
    n, cz_blocks = load_cz_blocks(qasm_path)
    row = math.ceil(math.sqrt(n))
    storage_flag = True#False

    list_gates = []
    for gates in cz_blocks:
        list_gates += powermove_stage_scheduling(gates, storage_flag)
    gate_stages = len(list_gates)

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            transfer_us, move_us, *_ = mvqc(cz_blocks, row, n, storage_flag, d, num_aod)
    else:
        transfer_us, move_us, *_ = mvqc(cz_blocks, row, n, storage_flag, d, num_aod)

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
            transfer_us, move_us, *_ = enola(cz_blocks, row, n, d)
    else:
        transfer_us, move_us, *_ = enola(cz_blocks, row, n, d)

    gate_time_us = gate_stages * T_RYDBERG_US
    transport_us = transfer_us + move_us

    return {
        "qasm": qasm_path,
        "qubits": n,
        "row": row,
        "d": d,
        "gate_stages": gate_stages,
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
    print(f"  gate_stages={result['gate_stages']}")
    print(f"  transfer_us={result['transfer_us']:.6f}")
    print(f"  move_us={result['move_us']:.6f}")
    print(f"  transport_us={result['transport_us']:.6f}")
    print(f"  gate_time_us={result['gate_time_us']:.6f}")
    print(f"  total_exec_us={result['total_exec_us']:.6f}")
    print(f"  t_comp_s={result['t_comp_s']:.6f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PowerMove then Enola on bb144_bivariate_bicycle_cz.qasm"
    )
    parser.add_argument(
        "--qasm",
        default=os.path.join(PROJECT_ROOT, "qasm_outputs", "bb144_bivariate_bicycle_cz.qasm"),
        help="Path to QASM file (default: qasm_outputs/bb144_bivariate_bicycle_cz.qasm)",
    )
    parser.add_argument("--d", type=int, default=1, help="Distance scaling factor d (default: 1)")
    parser.add_argument("--num-aod", type=int, default=1, help="PowerMove number of AODs (default: 1)")
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

    pm = run_powermove(qasm_path, args.d, args.num_aod, quiet)
    en = run_enola(qasm_path, args.d, quiet)

    print_result("PowerMove", pm)
    print_result("Enola", en)


if __name__ == "__main__":
    main()
