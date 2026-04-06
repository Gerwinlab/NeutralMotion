from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Iterable

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import extract_index_from_bit, format_node_line, op_node_signature
from .grid import Qubit

MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent


def _is_even_even(position: tuple[int, int]) -> bool:
    x, y = position
    return (x % 2 == 0) and (y % 2 == 0)


def _pulse_signature(node: DAGOpNode) -> str:
    """Return pulse identity (gate + params) used for pulse-switch accounting."""
    if node.op.name == "measure":
        return "measure"
    gate_name, gate_params, _ = op_node_signature(node)
    params_str = ",".join(str(param) for param in gate_params)
    return f"{gate_name}({params_str})" if params_str else gate_name


def schedule_single_qubit_time_steps(single_qubit_layer: list[DAGOpNode]) -> tuple[list[str], list[int]]:
    """Pack 1Q/measure nodes into timesteps while preserving per-qubit gate order."""
    if not single_qubit_layer:
        return [], []

    timesteps: list[list[str]] = []
    pulse_sets: list[set[str]] = []
    next_step_for_qubit: dict[int, int] = {}

    for node in single_qubit_layer:
        if len(node.qargs) != 1:
            continue
        qid = extract_index_from_bit(node.qargs[0])
        step_idx = next_step_for_qubit.get(qid, 0)

        while len(timesteps) <= step_idx:
            timesteps.append([])
            pulse_sets.append(set())

        timesteps[step_idx].append(format_node_line(node))
        pulse_sets[step_idx].add(_pulse_signature(node))
        next_step_for_qubit[qid] = step_idx + 1

    timestep_lines = [" ".join(step) for step in timesteps]
    unique_pulse_counts = [len(pulses) for pulses in pulse_sets]
    return timestep_lines, unique_pulse_counts


def single_qubit_layer_time(
    single_qubit_layer: list[DAGOpNode],
    average_single_gate_time: Any,
    t_switch: Any,
) -> tuple[list[str], Any]:
    """Compute scheduled single-qubit timesteps and total pulse time contribution."""
    timestep_lines, unique_pulse_counts = schedule_single_qubit_time_steps(single_qubit_layer)
    total_time = 0 * (average_single_gate_time + t_switch)
    for count in unique_pulse_counts:
        total_time += count * (average_single_gate_time + t_switch)
    return timestep_lines, total_time


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


def _format_move_action(event: MoveEvent) -> str | None:
    _, qubit_id, start_pos, end_pos = event
    if start_pos == end_pos:
        return None
    x1, y1 = start_pos
    x2, y2 = end_pos
    if _is_even_even(start_pos):
        return f"load q[{qubit_id}] -> ({x1},{y1}) : ({x2},{y2})"
    if _is_even_even(end_pos):
        return f"unload q[{qubit_id}] -> ({x1},{y1}) : ({x2},{y2})"
    return f"move q[{qubit_id}] ({x1},{y1}) : ({x2},{y2})"


def _move_batch_key(event: MoveEvent) -> tuple[str, tuple[int, int]]:
    _, _, start_pos, end_pos = event
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    if _is_even_even(start_pos):
        kind = "load"
    elif _is_even_even(end_pos):
        kind = "unload"
    else:
        kind = "move"
    return kind, (dx, dy)


def count_emitted_timesteps(events: Iterable[ScheduleEvent]) -> int:
    """Count emitted schedule timesteps with move batching semantics used by writer."""
    event_list = list(events)
    count = 0
    idx = 0
    while idx < len(event_list):
        event = event_list[idx]
        if event[0] == "gate":
            count += 1
            idx += 1
            continue
        move_event = event
        if _format_move_action(move_event) is None:
            idx += 1
            continue
        kind, delta = _move_batch_key(move_event)
        used_qubits = {move_event[1]}
        idx += 1
        while idx < len(event_list):
            nxt = event_list[idx]
            if nxt[0] == "gate":
                break
            if _format_move_action(nxt) is None:
                idx += 1
                continue
            nxt_kind, nxt_delta = _move_batch_key(nxt)
            if nxt_kind != kind or nxt_delta != delta or nxt[1] in used_qubits:
                break
            used_qubits.add(nxt[1])
            idx += 1
        count += 1
    return count


def write_timed_schedule(
    output_path: str | Path,
    *,
    solver: str,
    qasm_filename: str,
    final_time: str,
    lattice_spacing: str,
    fill_seed: int,
    events: Iterable[ScheduleEvent],
    initial_qubits: Iterable[Qubit] | None = None,
) -> None:
    """Write a gate-only textual schedule with optional initialization section."""
    out_path = Path(output_path)
    lines: list[str] = [
        f"solver: {solver}",
        f"qasm_file: {qasm_filename}",
        f"final_time: {final_time}",
        f"lattice_spacing (rydberg_radius): {lattice_spacing}",
        f"random number seed: {fill_seed}",
        "",
    ]

    init_line = _format_initialization_line(initial_qubits or [])
    if init_line:
        lines.append("T=0")
        lines.append(init_line)

    emitted = 0
    event_list = list(events)
    idx = 0
    while idx < len(event_list):
        event = event_list[idx]
        if event[0] == "gate":
            action = event[1]
            idx += 1
        else:
            move_action = _format_move_action(event)
            if move_action is None:
                idx += 1
                continue
            kind, delta = _move_batch_key(event)
            used_qubits = {event[1]}
            actions = [move_action]
            idx += 1
            while idx < len(event_list):
                nxt = event_list[idx]
                if nxt[0] == "gate":
                    break
                nxt_action = _format_move_action(nxt)
                if nxt_action is None:
                    idx += 1
                    continue
                nxt_kind, nxt_delta = _move_batch_key(nxt)
                if nxt_kind != kind or nxt_delta != delta or nxt[1] in used_qubits:
                    break
                used_qubits.add(nxt[1])
                actions.append(nxt_action)
                idx += 1
            action = " ".join(actions)
        emitted += 1
        t_step = emitted
        lines.append(f"T={t_step}")
        lines.append(action)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
