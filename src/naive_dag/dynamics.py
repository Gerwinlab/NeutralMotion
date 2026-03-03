from __future__ import annotations

from typing import Iterable

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import op_node_signature
from .grid import GridNode, Qubit

#TODO: Fix what Codex wrote with actually working code that is good. Essentially, look at next step and if next step includes the ith qubit then move there. Otherwise store.
#TODO: There should be at most 1 turn. But it should aim at going to a good site. Decide final site if qubit is store then where to transfer to aod.
#TODO: if the qubit is in the highway after a gate op, then decide where to go based on initial. Straight paths only.
def gate_qubit_ids(ops: list[DAGOpNode], gate_index: int) -> list[int]:
    """Return the qubit ids used by the ith gate in ops."""
    _, _, qubit_ids = op_node_signature(ops[gate_index])
    if len(qubit_ids) > 2:
        raise ValueError("This method does not support 3+ qubit gates")
    return qubit_ids


def gate_qubits(ops: list[DAGOpNode], gate_index: int, qubits: list[Qubit]) -> list[Qubit]:
    """Resolve qubit objects for the ith gate in ops."""
    qubit_ids = gate_qubit_ids(ops, gate_index)
    qubit_map = {q.id: q for q in qubits}
    return [qubit_map[qid] for qid in qubit_ids]


def _in_bounds(grid: list[list[GridNode]], row: int, col: int) -> bool:
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def _start_anchors(row: int, col: int, grid: list[list[GridNode]]) -> list[tuple[int, int]]:
    if row % 2 == 0 and col % 2 == 0:
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        return [(r, c) for r, c in candidates if _in_bounds(grid, r, c)]
    return [(row, col)]


def _end_sites(row: int, col: int, grid: list[list[GridNode]]) -> list[tuple[int, int]]:
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if _in_bounds(grid, r, c)]


def _line_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    sr, sc = start
    er, ec = end
    if sr == er:
        step = 1 if ec >= sc else -1
        return [(sr, c) for c in range(sc, ec + step, step)]
    if sc == ec:
        step = 1 if er >= sr else -1
        return [(r, sc) for r in range(sr, er + step, step)]
    return None


def _path_on_odd_line(path: list[tuple[int, int]]) -> bool:
    if not path:
        return False
    if len(path) == 1:
        r, c = path[0]
        return (r % 2 == 1) or (c % 2 == 1)
    r0, c0 = path[0]
    r1, c1 = path[1]
    if r0 == r1:
        return r0 % 2 == 1
    if c0 == c1:
        return c0 % 2 == 1
    return False


def _two_segment_path(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if start == end:
        return [start]

    sr, sc = start
    er, ec = end
    pivots = [(sr, ec), (er, sc)]
    best: list[tuple[int, int]] | None = None

    for pr, pc in pivots:
        first = _line_path(start, (pr, pc))
        second = _line_path((pr, pc), end)
        if first is None or second is None:
            continue
        if not _path_on_odd_line(first) or not _path_on_odd_line(second):
            continue
        path = first + second[1:]
        if best is None or len(path) < len(best):
            best = path
    return best


def best_path_for_gate(
    ops: list[DAGOpNode],
    gate_index: int,
    qubits: list[Qubit],
    grid: list[list[GridNode]],
) -> list[tuple[int, int]]:
    """
    Find the shortest valid path for the first qubit to reach a neighbor of the second qubit.

    Movement rules:
    - If the first qubit starts on an even-even node, it must first move to an adjacent node.
    - The long move must be a straight line along an odd row or odd column.
    - The destination must be one of the 4 neighbor sites of the second qubit.
    """
    q1, q2 = gate_qubits(ops, gate_index, qubits)
    r1, c1 = q1.grid_position()
    r2, c2 = q2.grid_position()

    starts = _start_anchors(r1, c1, grid)
    targets = _end_sites(r2, c2, grid)

    best: list[tuple[int, int]] | None = None
    for start in starts:
        for target in targets:
            line = _two_segment_path(start, target)
            if line is None:
                continue
            prefix: list[tuple[int, int]] = [(r1, c1)] if start != (r1, c1) else []
            if start != (r1, c1):
                prefix.append(start)
            path = prefix + line[1:] if prefix else line
            if best is None or len(path) < len(best):
                best = path

    if best is None:
        raise ValueError("No valid straight-line path found for this gate.")
    return best
