from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import op_node_signature
from .grid import Qubit


MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent


def _is_even_even(position: tuple[int, int]) -> bool:
    """Return True when both grid coordinates are even."""
    x, y = position
    return (x % 2 == 0) and (y % 2 == 0)


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


def _format_gate_line(gate_name: str, gate_params: list, qubit_ids: list[int]) -> str:
    """Build a single OpenQASM-like gate line from normalized gate fields."""
    if gate_params:
        params_str = ", ".join(_format_gate_param(param) for param in gate_params)
        return f"{gate_name}({params_str}) " + ",".join(f"q[{qid}]" for qid in qubit_ids) + ";"
    return f"{gate_name} " + ",".join(f"q[{qid}]" for qid in qubit_ids) + ";"


def collect_single_qubit_gate_block(ops: list[DAGOpNode], start_index: int) -> tuple[list[str], int, list[int]]:
    """Collect contiguous 1Q gates and pack them into disjoint-qubit time layers.

    The scan starts at ``start_index`` and stops before the next 2Q gate.
    Returns:
    - A list of gate lines, one line per time layer.
    - The index where scanning stopped.
    - The number of 1Q pulses in each returned layer.
    """
    layers: list[list[str]] = []
    layer_qubits: list[set[int]] = []
    layer_counts: list[int] = []
    i = start_index
    while i < len(ops):
        gate_name, gate_params, qubit_ids = op_node_signature(ops[i])
        if len(qubit_ids) == 2:
            break
        if len(qubit_ids) == 1:
            qid = qubit_ids[0]
            gate_line = _format_gate_line(gate_name, gate_params, qubit_ids)
            placed = False
            for layer_idx, used_qubits in enumerate(layer_qubits):
                if qid not in used_qubits:
                    layers[layer_idx].append(gate_line)
                    used_qubits.add(qid)
                    layer_counts[layer_idx] += 1
                    placed = True
                    break
            if not placed:
                layers.append([gate_line])
                layer_qubits.append({qid})
                layer_counts.append(1)
        i += 1
    return [" ".join(layer) for layer in layers], i, layer_counts


def _format_initialization_line(qubits: Iterable[Qubit]) -> str:
    """Render a deterministic initialization statement for all provided qubits."""
    ordered = sorted(qubits, key=lambda qubit: qubit.id)
    entries = []
    for qubit in ordered:
        row, col = qubit.grid_position()
        entries.append(f"q[{qubit.id}] -> ({row},{col})")
    if not entries:
        return ""
    return "initialize " + "; ".join(entries) + ";"


def write_timed_schedule(
    output_path: str | Path,
    *,
    qasm_filename: str,
    final_time: str,
    lattice_spacing: str,
    T: int,
    fill_seed:int,
    events: Iterable[ScheduleEvent],
    initial_qubits: Iterable[Qubit] | None = None,
) -> None:
    """Write the textual schedule with optional initialization and timed events.

    Move events with identical start and end positions are omitted.
    """
    out_path = Path(output_path)
    lines: list[str] = [
        f"qasm_file: {qasm_filename}",
        f"final_time: {final_time}",
        f"lattice_spacing (rydberg_radius): {lattice_spacing}",
        f"random number seed: {fill_seed}"
        "",
    ]

    init_line = _format_initialization_line(initial_qubits or [])
    if init_line:
        lines.append("T=0")
        lines.append(init_line)

    emitted = 0
    for event in events:
        if event[0] == "gate":
            action = event[1]
        else:
            _, qubit_id, start_pos, end_pos = event
            if start_pos == end_pos:
                # Skip no-op moves so they do not consume a printed time step.
                continue
            x1, y1 = start_pos
            x2, y2 = end_pos
            if _is_even_even(start_pos):
                action = f"load q[{qubit_id}] -> ({x1},{y1}) : ({x2},{y2})"
            elif _is_even_even(end_pos):
                action = f"unload q[{qubit_id}] -> ({x1},{y1}) : ({x2},{y2})"
            else:
                action = f"move q[{qubit_id}] ({x1},{y1}) : ({x2},{y2})"
        emitted += 1
        t_step = T + emitted
        lines.append(f"T={t_step}")
        lines.append(action)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
