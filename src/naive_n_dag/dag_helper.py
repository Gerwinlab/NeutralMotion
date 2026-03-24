from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode


from qiskit.circuit.library.standard_gates import SwapGate


def load_qasm_to_circuit(qasm_path: str | Path) -> QuantumCircuit:
    """Load a QASM file into a QuantumCircuit."""
    return QuantumCircuit.from_qasm_file(str(qasm_path))


def load_qasm_to_dag(qasm_path: str | Path) -> DAGCircuit:
    """Load a QASM file and convert it to a DAGCircuit."""
    circuit = load_qasm_to_circuit(qasm_path)
    return circuit_to_dag(circuit)


def dag_with_gate_ops_only(dag: DAGCircuit) -> list[DAGOpNode]:
    """Return topologically ordered op-nodes with SWAP operations removed."""
    gate_only = dag.copy_empty_like()

    for node in dag.topological_op_nodes():
        if isinstance(node.op, SwapGate):
            continue
        gate_only.apply_operation_back(node.op, node.qargs, node.cargs)

    return gate_only.op_nodes()

#Work on the n dags. Will reuse dynamic solver from naive_dag.
def load_qasm_to_gate_dag(qasm_path: str | Path) -> DAGCircuit:
    """Load a QASM file and return a DAGCircuit with only gate operations."""
    return dag_with_gate_ops_only(load_qasm_to_dag(qasm_path))


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
