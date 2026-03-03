from __future__ import annotations

from pathlib import Path

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import op_node_signature
from .dynamics import best_path_for_gate, gate_qubits
from .grid import GridNode, Qubit

#TODO: Fix what codex write into actual target code. Codex just produces some skeleton code for me.
def write_schedule_like_qasm(
    ops: list[DAGOpNode],
    qubits: list[Qubit],
    grid: list[list[GridNode]],
    output_path: str | Path,
) -> None:
    """
    Write a simple, qasm-like schedule:
    - movement lines with path in (r,c) coordinates
    - gate lines with qubit ids and params
    """
    out_path = Path(output_path)
    lines: list[str] = []

    for idx in range(len(ops)):
        gate, params, qubit_ids = op_node_signature(ops[idx])
        if len(qubit_ids) > 2:
            raise ValueError("This method does not support 3+ qubit gates")

        q1, q2 = gate_qubits(ops, idx, qubits)
        r1, c1 = q1.grid_position()
        r2, c2 = q2.grid_position()

        path = best_path_for_gate(ops, idx, qubits, grid)
        path_str = " -> ".join(f"({r},{c})" for r, c in path)
        lines.append(f"move q{q1.id} from ({r1},{c1}) along {path_str} to neighbor of q{q2.id} ({r2},{c2});")

        if params:
            params_str = ", ".join(f"{p:g}" for p in params)
            lines.append(f"{gate}({params_str}) q[{qubit_ids[0]}], q[{qubit_ids[1]}];")
        else:
            lines.append(f"{gate} q[{qubit_ids[0]}], q[{qubit_ids[1]}];")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
