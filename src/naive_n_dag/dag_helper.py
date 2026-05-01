from __future__ import annotations

from pathlib import Path
import math
import re
from fractions import Fraction

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode


from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.circuit.library.standard_gates import CXGate


def load_qasm_to_circuit(qasm_path: str | Path) -> QuantumCircuit:
    """Load a QASM file into a QuantumCircuit."""
    return QuantumCircuit.from_qasm_file(str(qasm_path))


def load_qasm_to_dag(qasm_path: str | Path) -> DAGCircuit:
    """Load a QASM file and convert it to a DAGCircuit."""
    circuit = load_qasm_to_circuit(qasm_path)
    return circuit_to_dag(circuit)

#Replaced by _extract_index_from_bit, may be useful for future reference:
def op_node_signature(node: DAGOpNode) -> tuple[str, list[float], list[int]]:
    """
    Return (gate_name, params, qubit_indices) for a DAGOpNode.

    Qubit indices are extracted from node.qargs (e.g., q[3] -> 3).
    """
    gate_name = node.op.name
    params = [float(p) for p in getattr(node.op, "params", [])]
    qubit_indices: list[int] = []
    for q in node.qargs:
        rep = repr(q)
        if "index=" not in rep:
            raise ValueError(f"Unable to extract qubit index from {rep}")
        idx_str = rep.split("index=", 1)[1].split(">", 1)[0].strip()
        idx = int(idx_str)
        qubit_indices.append(int(idx))
    return gate_name, params, qubit_indices


def _format_gate_param(param) -> str:
    """Format a gate parameter with compact pi-based notation when possible."""
    try:
        value = float(param)
    except (TypeError, ValueError):
        return str(param)

    if math.isclose(value, 0.0, abs_tol=1e-12):
        return "0"

    ratio = value / math.pi
    frac = Fraction(ratio).limit_denominator(32)
    if math.isclose(ratio, float(frac), rel_tol=0, abs_tol=1e-10):
        n, d = frac.numerator, frac.denominator
        if d == 1:
            if n == 1:
                return "pi"
            if n == -1:
                return "-pi"
            return f"{n}*pi"
        if n == 1:
            return f"pi/{d}"
        if n == -1:
            return f"-pi/{d}"
        return f"{n}*pi/{d}"

    return str(param)


def format_gate_line(gate_name: str, gate_params: list, qubit_ids: list[int]) -> str:
    """Build a single OpenQASM-like gate line from normalized gate fields."""
    if gate_params:
        params_str = ", ".join(_format_gate_param(param) for param in gate_params)
        return f"{gate_name}({params_str}) " + ",".join(f"q[{qid}]" for qid in qubit_ids) + ";"
    return f"{gate_name} " + ",".join(f"q[{qid}]" for qid in qubit_ids) + ";"


def extract_index_from_bit(bit) -> int:
    """Extract the integer index from a Qiskit bit repr (e.g. Qubit(index=3))."""
    rep = repr(bit)
    if "index=" not in rep:
        raise ValueError(f"Unable to extract bit index from {rep}")
    idx_str = rep.split("index=", 1)[1].split(">", 1)[0].strip()
    return int(idx_str)


def format_node_line(node: DAGOpNode) -> str:
    """Render a DAG op node as a schedule line, including measurement formatting."""
    qubit_indices = [extract_index_from_bit(q) for q in node.qargs]
    if node.op.name == "measure":
        if len(node.qargs) != 1 or len(node.cargs) != 1:
            raise ValueError("Measurement node must have exactly one qarg and one carg.")
        classical_idx = extract_index_from_bit(node.cargs[0])
        return f"measure q[{qubit_indices[0]}] -> c[{classical_idx}];"

    params = [float(p) for p in getattr(node.op, "params", [])]
    return format_gate_line(node.op.name, params, qubit_indices)


def build_two_qubit_only_dag_with_single_qubit_context(
    dag: DAGCircuit,
) -> tuple[DAGCircuit, list[list[DAGOpNode]]]:
    """Return a 2Q-only DAG and non-2Q ops grouped by preceding 2Q layer boundary.

    The returned tuple is:
    - `two_qubit_dag`: DAG containing only 2Q operations.
    - `single_layers`: list whose length is `len(list(two_qubit_dag.layers())) + 1`.
      `single_layers[i]` holds lines to run before two-qubit layer `i`, and the
      last entry holds lines to run after the final two-qubit layer.

    Measurements are included in `single_layers`.
    """
    two_qubit_dag = dag.copy_empty_like()
    # For each qubit, track the next 2Q layer index after the latest 2Q op touching it.
    qubit_twoq_progress: dict[int, int] = {}
    num_twoq_layers = 0
    single_layers: list[list[DAGOpNode]] = [[]]

    for node in dag.topological_op_nodes():
        if isinstance(node.op, SwapGate):
            continue

        num_qargs = len(node.qargs)
        if num_qargs == 2:
            q0 = extract_index_from_bit(node.qargs[0])
            q1 = extract_index_from_bit(node.qargs[1])
            layer_idx = max(qubit_twoq_progress.get(q0, 0), qubit_twoq_progress.get(q1, 0))
            qubit_twoq_progress[q0] = layer_idx + 1
            qubit_twoq_progress[q1] = layer_idx + 1
            num_twoq_layers = max(num_twoq_layers, layer_idx + 1)
            two_qubit_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            continue

        if num_qargs == 1:
            qid = extract_index_from_bit(node.qargs[0])
            bucket_idx = qubit_twoq_progress.get(qid, 0)
            while len(single_layers) <= bucket_idx:
                single_layers.append([])
            single_layers[bucket_idx].append(node)

    while len(single_layers) < num_twoq_layers + 1:
        single_layers.append([])

    return two_qubit_dag, single_layers


def load_qasm_to_two_qubit_dag_with_single_qubit_context(
    qasm_path: str | Path,
) -> tuple[DAGCircuit, list[list[DAGOpNode]]]:
    """Load QASM and build a 2Q-only DAG plus per-layer single-qubit context."""
    return build_two_qubit_only_dag_with_single_qubit_context(load_qasm_to_dag(qasm_path))


_TXT_GATE_LINE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s+(\d+)\s*$")
_TXT_SUMMARY_LINE_RE = re.compile(r"^\s*T\s*=\s*(\d+)\s*,\s*cx\s*=\s*(\d+)\s*$")


def dag_from_txt_auto(txt_path: str | Path) -> tuple[DAGCircuit, list[list[DAGOpNode]]]:
    """Build a two-qubit DAG from a timestep text file.

    Expected format per timestep:
    - one or more gate lines: ``<T>, <gate_index>: <q0> <q1>``
    - followed by summary line: ``T = <T>, cx = <count>``

    Example:
    ``4, 4: 1 2``
    ``4, 6: 0 3``
    ``T = 4, cx = 2``
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"txt gate-order file not found: {path}")

    steps: list[list[tuple[int, int]]] = []
    current_timestep: int | None = None
    current_pairs: list[tuple[int, int]] = []
    expected_next_timestep: int | None = None
    max_qubit_id = -1
    saw_content = False

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            saw_content = True

            gate_match = _TXT_GATE_LINE_RE.match(line)
            if gate_match is not None:
                timestep, _gate_index, q0, q1 = (int(v) for v in gate_match.groups())
                if q0 == q1:
                    raise ValueError(
                        f"{path}:{line_no}: invalid two-qubit gate uses the same qubit id twice: {q0}."
                    )

                if current_timestep is None:
                    if expected_next_timestep is not None and timestep != expected_next_timestep:
                        raise ValueError(
                            f"{path}:{line_no}: expected timestep {expected_next_timestep}, got {timestep}."
                        )
                    current_timestep = timestep
                    current_pairs = []
                elif timestep != current_timestep:
                    raise ValueError(
                        f"{path}:{line_no}: encountered timestep {timestep} before summary line for "
                        f"timestep {current_timestep}."
                    )

                current_pairs.append((q0, q1))
                max_qubit_id = max(max_qubit_id, q0, q1)
                continue

            summary_match = _TXT_SUMMARY_LINE_RE.match(line)
            if summary_match is not None:
                summary_timestep, cx_count = (int(v) for v in summary_match.groups())
                if current_timestep is None:
                    raise ValueError(
                        f"{path}:{line_no}: found summary line before any gate lines for timestep {summary_timestep}."
                    )
                if summary_timestep != current_timestep:
                    raise ValueError(
                        f"{path}:{line_no}: summary timestep {summary_timestep} does not match "
                        f"gate timestep {current_timestep}."
                    )
                if cx_count != len(current_pairs):
                    raise ValueError(
                        f"{path}:{line_no}: summary cx={cx_count} does not match {len(current_pairs)} gate "
                        f"line(s) for timestep {current_timestep}."
                    )

                steps.append(current_pairs)
                expected_next_timestep = current_timestep + 1
                current_timestep = None
                current_pairs = []
                continue

            raise ValueError(
                f"{path}:{line_no}: invalid format. Expected '<T>, <gate_index>: <q0> <q1>' or "
                "'T = <T>, cx = <count>'."
            )

    if not saw_content:
        raise ValueError(f"{path}: file is empty; expected timestep gate lines.")
    if current_timestep is not None:
        raise ValueError(
            f"{path}: missing summary line for final timestep {current_timestep}."
        )
    if not steps:
        raise ValueError(f"{path}: no timestep blocks parsed from file.")

    circuit = QuantumCircuit(max_qubit_id + 1)
    two_qubit_dag = circuit_to_dag(circuit)
    qubits = two_qubit_dag.qubits
    for timestep_pairs in steps:
        for q0, q1 in timestep_pairs:
            two_qubit_dag.apply_operation_back(CXGate(), qargs=[qubits[q0], qubits[q1]], cargs=[])

    single_layers: list[list[DAGOpNode]] = [[] for _ in range(len(list(two_qubit_dag.layers())) + 1)]
    return two_qubit_dag, single_layers
