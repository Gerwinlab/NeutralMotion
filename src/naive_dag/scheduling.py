from __future__ import annotations

from pathlib import Path
from typing import Iterable


MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent


def _is_even_even(position: tuple[int, int]) -> bool:
    x, y = position
    return (x % 2 == 0) and (y % 2 == 0)
#TODO: Add timing for switching between pulses with a t_switch of 1.5 microseconds and max gate time of .5 microseconds for 2 qubit gate.
#TODO: Add timing for average single qubti gate and two_qubit gate. single qubit gate could be .1 microseconds
#TODO: Fix issue of adding time step for no movement.


def write_timed_schedule(
    output_path: str | Path,
    *,
    qasm_filename: str,
    final_time: str,
    lattice_spacing: str,
    T: int,
    events: Iterable[ScheduleEvent],
) -> None:
    out_path = Path(output_path)
    lines: list[str] = [
        f"qasm_file: {qasm_filename}",
        f"final_time: {final_time}",
        f"lattice_spacing (rydberg_radius): {lattice_spacing}",
        "",
    ]

    for offset, event in enumerate(events, start=1):
        t_step = T + offset
        if event[0] == "gate":
            action = event[1]
        else:
            _, qubit_id, start_pos, end_pos = event
            x1, y1 = start_pos
            x2, y2 = end_pos
            if _is_even_even(start_pos):
                action = f"load q[{qubit_id}] -> ({x1},{y1}) : ({x2},{y2})"
            elif _is_even_even(end_pos):
                action = f"unload q[{qubit_id}] -> ({x1},{y1}) : ({x2},{y2})"
            else:
                action = f"move q[{qubit_id}] ({x1},{y1}) : ({x2},{y2})"
        lines.append(f"T={t_step}")
        lines.append(action)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
