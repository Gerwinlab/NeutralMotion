from __future__ import annotations

from pathlib import Path

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Barrier
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode


def load_qasm_to_circuit(qasm_path: str | Path) -> QuantumCircuit:
    """Load a QASM file into a QuantumCircuit."""
    return QuantumCircuit.from_qasm_file(str(qasm_path))


def ancilla_indices(circuit: QuantumCircuit, names) -> frozenset[int]:
    """Global indices of the qubits in the QASM registers ``names`` (``ancilla_registers``)."""
    registers = {register.name: register for register in circuit.qregs}
    missing = sorted(set(names) - registers.keys())
    if missing:
        raise ValueError(f"ancilla_registers {missing} are not declared in the QASM (registers: {sorted(registers)}).")
    return frozenset(circuit.find_bit(q).index for name in names for q in registers[name])


def prepare_circuit(
    circuit: QuantumCircuit, optimization_level: int = 0, basis_gates: list[str] | None = None,
    ancilla_registers=(),
) -> QuantumCircuit:
    """Transpile unless ``optimization_level`` is 0 with no basis; record ancillas.

    Level 0 without a basis would only reorder independent gates, so the
    circuit is kept as written. Transpile settings go to
    ``metadata["transpile"]`` and the qubits of ``ancilla_registers`` to
    ``metadata["ancilla_qubits"]``.
    """
    ancillas = ancilla_indices(circuit, ancilla_registers)
    metadata = dict(circuit.metadata or {})
    if optimization_level or basis_gates is not None:
        circuit = transpile(circuit, basis_gates=basis_gates,
                            optimization_level=optimization_level, seed_transpiler=0)
        metadata["transpile"] = (optimization_level, basis_gates)
    if ancillas:
        metadata["ancilla_qubits"] = sorted(ancillas)
    circuit.metadata = metadata
    return circuit


def flatten_quantum_registers(dag: DAGCircuit) -> DAGCircuit:
    """Merge several quantum registers into one register ``q``, in declaration order.

    The solver identifies a qubit by its index (``extract_index_from_bit``),
    which is local to its register, so ``l[5]`` and ``r[5]`` would collide.
    Qubit ``reg[i]`` becomes ``q[start + i]``, where ``start`` is the number
    of qubits declared before ``reg``: Qiskit's global bit index. Classical
    registers are kept, so measurements still name their own register
    (``measure q[126] -> sx[0];``). ``metadata["qubit_registers"]`` records
    ``(name, start, size)`` per original register for the schedule header.
    A DAG with at most one quantum register is returned unchanged.
    """
    registers = list(dag.qregs.values())
    if len(registers) <= 1:
        return dag
    if sum(register.size for register in registers) != dag.num_qubits():
        raise ValueError("Every qubit must belong to exactly one quantum register.")
    flat = DAGCircuit()
    flat.name, flat.global_phase = dag.name, dag.global_phase
    flat.metadata = dict(dag.metadata or {})
    register = QuantumRegister(dag.num_qubits(), "q")
    flat.add_qreg(register)
    for creg in dag.cregs.values():
        flat.add_creg(creg)
    loose = [bit for bit in dag.clbits if not dag.find_bit(bit).registers]
    if loose:
        flat.add_clbits(loose)
    qubit = {bit: register[dag.find_bit(bit).index] for bit in dag.qubits}
    for node in sorted(dag.op_nodes(), key=lambda item: item._node_id):
        flat.apply_operation_back(node.op, [qubit[bit] for bit in node.qargs], node.cargs)
    flat.metadata["qubit_registers"] = [
        (reg.name, dag.find_bit(reg[0]).index, reg.size) for reg in registers
    ]
    return flat


def format_qubit_registers(layout) -> str | None:
    """``l=q[0:63] r=q[63:126] ...`` for a flattened DAG's register layout, else None."""
    if not layout:
        return None
    return " ".join(f"{name}=q[{start}:{start + size}]" for name, start, size in layout)


def ancilla_qubits(dag: DAGCircuit) -> frozenset[int]:
    """The qubits of ``ancilla_registers`` recorded by ``prepare_circuit`` (they keep the role all circuit)."""
    return frozenset((dag.metadata or {}).get("ancilla_qubits", ()))


def format_transpile(metadata) -> str | None:
    """``optimization_level=2 basis_gates=u3,cz`` when ``prepare_circuit`` transpiled, else None."""
    if not metadata or "transpile" not in metadata:
        return None
    level, basis = metadata["transpile"]
    return f"optimization_level={level} basis_gates={','.join(basis) if basis else 'as written'}"


def load_qasm_to_dag(
    qasm_path: str | Path, optimization_level: int = 0, basis_gates: list[str] | None = None,
    ancilla_registers=(),
) -> DAGCircuit:
    """Load a prepared QASM circuit (``prepare_circuit``) as a DAG with one quantum register."""
    circuit = prepare_circuit(load_qasm_to_circuit(qasm_path), optimization_level, basis_gates, ancilla_registers)
    return flatten_quantum_registers(circuit_to_dag(circuit))


def dag_with_gate_ops_only(dag: DAGCircuit) -> list[DAGOpNode]:
    """Return source-ordered op nodes with every barrier widened to all qubits."""
    gate_only = dag.copy_empty_like()

    for node in sorted(dag.op_nodes(), key=lambda item: item._node_id):
        if node.op.name == "barrier":
            gate_only.apply_operation_back(Barrier(len(dag.qubits)), dag.qubits, [])
        else:
            gate_only.apply_operation_back(node.op, node.qargs, node.cargs)

    return gate_only.op_nodes()

#TODO: This way of loading and making a dag can be limiting. Can do 1million+ gates but needs modification for future use.
def load_qasm_to_gate_dag(
    qasm_path: str | Path, optimization_level: int = 0, basis_gates: list[str] | None = None,
    ancilla_registers=(),
) -> DAGCircuit:
    """Load a QASM file and return its op nodes as filtered by ``dag_with_gate_ops_only``."""
    return dag_with_gate_ops_only(load_qasm_to_dag(qasm_path, optimization_level, basis_gates, ancilla_registers))


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


def format_classical_bit(bit) -> str:
    """Render a classical bit with its QASM register name and local index."""
    register = getattr(bit, "_register", None)
    index = getattr(bit, "_index", None)
    if register is not None and index is not None:
        return f"{register.name}[{int(index)}]"

    rep = repr(bit)
    if "index=" not in rep:
        raise ValueError(f"Unable to extract classical bit index from {rep}")
    index = int(rep.split("index=", 1)[1].split(">", 1)[0].strip())
    return f"c[{index}]"
