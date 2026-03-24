from __future__ import annotations

from qiskit.dagcircuit.dagnode import DAGOpNode

from .dag_helper import op_node_signature
from .grid import GridNode, Qubit, move_qubit
from .scheduling import _format_gate_line, collect_single_qubit_gate_block

MoveEvent = tuple[str, int, tuple[int, int], tuple[int, int]]
GateEvent = tuple[str, str]
ScheduleEvent = MoveEvent | GateEvent

def gate_qubit_ids(ops: list[DAGOpNode], gate_index: int) -> list[int]:
    """Return qubit ids used by the operation at ``gate_index``."""
    _, _, qubit_ids = op_node_signature(ops[gate_index])
    if len(qubit_ids) > 2:
        raise ValueError("This method does not support 3+ qubit gates")
    return qubit_ids


def gate_qubits(ops: list[DAGOpNode], gate_index: int, qubits: list[Qubit]) -> list[Qubit]:
    """Map gate qubit ids to concrete ``Qubit`` objects from ``qubits``."""
    qubit_ids = gate_qubit_ids(ops, gate_index)
    if not qubit_ids or len(qubit_ids) < 2:
        return [None,None]
    qubit_map = {q.id: q for q in qubits}
    return [qubit_map[qid] for qid in qubit_ids]


def _in_bounds(grid: list[list[GridNode]], row: int, col: int) -> bool:
    """Return True if ``(row, col)`` is within the rectangular grid bounds."""
    return 0 <= row < len(grid) and 0 <= col < len(grid[0])


def _start_(q1: Qubit, q2: Qubit, moves: list[tuple[int, int]], grid: list[list[GridNode]]) -> None:
    """Append the first move that places ``q1`` onto a valid movement highway."""
    row1, col1 = q1.grid_position()
    row2, col2 = q2.grid_position()
    q1_odd = (row1 % 2 != 0) or (col1 % 2 != 0)
    if q1_odd != True: #if it is in a grid position
        if row1 == row2:
            if col2-col1==2:
                moves.append((row1,col2-1))
            elif col1-col2==2:
                moves.append((row1,col2+1))
            elif _in_bounds(grid,row1+1,col1):
                moves.append((row1+1,col1))
            else:
                moves.append((row1-1,col1))
        elif col1 == col2:
            if row2-row1==2:
                moves.append((row1-1,col1))
            elif row1-row2==2:
                moves.append((row1+1,col1))
            elif _in_bounds(grid,row1,col1+1):
                moves.append((row1,col1+1))
            else:
                moves.append((row1,col1-1))
        elif abs(row1-row2) > abs(col1-col2):
            if col1-col2 > 2:
                moves.append((row1,col1+1))
            else:
                moves.append((row1,col1-1))
        else:
            if row1-row2 > 2:
                moves.append((row1+1,col1))
            else:
                moves.append((row1-1,col1))

def _shuttle_(q1_pos: tuple[int, int], q2_pos: tuple[int, int], moves: list[tuple[int, int]]) -> None:
    """Append intermediate straight-line routing moves from ``q1_pos`` toward ``q2_pos``."""
    row1, col1 = q1_pos
    row2, col2 = q2_pos

    if row1 == row2 + 1 or row1 == row2 - 1:
        moves.append((row1,col2))
    elif col1 == col2 + 1 or col2 == col2 - 1:
        moves.append((row2,col1))
    elif abs(row2-row1) > abs(col2-col1):
        if row2 - row1 > 0:
            moves.append(((row2 - 1),col1))
            moves.append(((row2 - 1),col2))
        else:
            moves.append(((row2 + 1),col1))
            moves.append(((row2 + 1),col2))
    else:
        if col2 - col1 > 0:
            moves.append((row1,col2-1))
            moves.append((row2,col2-1))
        else:
            moves.append((row1,col2+1))
            moves.append((row2,col2+1))

def _return_(q1:Qubit, moves: list[tuple[int, int]], grid: list[list[GridNode]]) -> None:
    """Route ``q1`` from highway space back to an available even-even trap site."""
    row1, col1 = q1.grid_position()
    q1_odd = (row1 % 2 != 0) or (col1 % 2 != 0)
    if not q1_odd:
        _shuttle_(moves[-1],q1.grid_position(),moves)
        moves.append(q1.grid_position())
    else:
        found = False
        for i in range(0, len(grid),2):
            for j in range(0,len(grid[i]),2):
                if not grid[i][j].is_occupied():
                    move_qubit(q1, grid[i][j])
                    found = True
                    break
            if found:
                break
        if not found:
            raise RuntimeError("No free even-position grid node found")
        _shuttle_(moves[-1],q1.grid_position(),moves)
        moves.append(q1.grid_position())

def find_next_two_qubit_gate(ops, start_index):
    """Return the index of the next 2Q gate after ``start_index``, else ``None``."""
    i = start_index + 1
    while i < len(ops):
        node = ops[i]
        if len(node.qargs) == 2:
            return i
        i += 1
    return None

def best_path_for_gate(
    ops: list[DAGOpNode],
    gate_index: int,
    qubits: list[Qubit],
    grid: list[list[GridNode]],
    config:dict,
    T: int
):
    """Plan movement/events for one 2Q gate and any immediate trailing 1Q block.

    Movement rules:
    - If the first qubit starts on an even-even node, it must first move to an adjacent node.
    - The long move must be a straight line along an odd row or odd column.
    - The destination must be one of the 4 neighbor sites of the second qubit.

    Returns the timestep duration contribution, updated timestep counter, and
    emitted schedule events for moves and gates.
    """
    Stay_in_Highway = False
    q1, q2 = gate_qubits(ops, gate_index, qubits)
    if q1 is None:
        return 0 * config["t_switch"], T, []
    r1, c1 = q1.grid_position()
    r2, c2 = q2.grid_position()
    #Checking which qubit to move
    q1_odd = (r1 % 2 != 0) or (c1 % 2 != 0)
    q2_odd = (r2 % 2 != 0) or (c2 % 2 != 0)
    i = find_next_two_qubit_gate(ops,gate_index)
    if q1_odd and q2_odd:
        raise ValueError("Both Atoms have odd positions - they shouldn't move at the same time.")
    elif q1_odd:
        q1, q2 = q1, q2
    elif q2_odd:
        q1, q2 = q2, q1 #swap so q1 is being moved
    else:#Checking which qubit to move based on if it is in another two gate
        if i == None:
            q1, q2 = q1, q2
        else:
            q3, q4 = gate_qubits(ops, i, qubits)
            if q2 == q3 or q2 == q4:
                q1, q2 = q2, q1
            else:
                q1, q2 = q1, q2
    if i != None: #if the qubit must move to another sight
        q3, q4 = gate_qubits(ops, i, qubits)
        if q1 == q3 or q1 == q4:
            Stay_in_Highway = True
    #-------------------------------------------
    # Finished with deciding which moves and how ends
    #-------------------------------------------
    moves = [q1.grid_position()]# This will hold how many moves, the length is the number of time steps taken
    events: list[ScheduleEvent] = []
    #Now if the qubit is not in a highway we must transfer to an AOD
    _start_(q1,q2,moves,grid)
    _shuttle_(moves[-1],q2.grid_position(),moves)

    for idx in range(len(moves) - 1):
        events.append(("move", q1.id, moves[idx], moves[idx + 1]))

    gate_name, gate_params, qubit_ids = op_node_signature(ops[gate_index])
    gate_line = _format_gate_line(gate_name, gate_params, qubit_ids)
    events.append(("gate", gate_line))
    one_qubit_layers, _, layer_counts = collect_single_qubit_gate_block(ops, gate_index + 1)
    for layer_line in one_qubit_layers:
        events.append(("gate", layer_line))

    split_index = len(moves) - 1
    if Stay_in_Highway:
        i, j = moves[-1]
        move_qubit(q1,grid[i][j])
    else:
        _return_(q1,moves,grid)
        for idx in range(split_index, len(moves) - 1):
            events.append(("move", q1.id, moves[idx], moves[idx + 1]))
    T_step = len(events)
    time = _time_trapezoid_(q1,q2,moves,config) + config["average_two_gate_time"] + config["t_switch"]
    # Single-qubit pulses always include a pulse-switch penalty per gate pulse.
    for layer_count in layer_counts:
        time += layer_count * (config["average_single_gate_time"] + config["t_switch"])

    return time, T_step + T, events

#--------------------------------
#Now for timing
#--------------------------------
def _time_trapezoid_(
    q1: Qubit,
    q2: Qubit,
    moves: list[tuple[int, int]],
    config:dict,
):
    """Compute movement time using a trapezoidal/triangular velocity profile."""
    v = config["max_velocity"]        # Pint Quantity
    a = config["max_acceleration"]    # Pint Quantity
    transfer_time = config["transfer_SLM_AOD"]
    grid_spacing = config["rydberg_radius"]

    total_time = 0 * (v / a).units    # initializes time quantity (seconds)

    for i in range(len(moves) - 1):
        x1, y1 = moves[i]
        x2, y2 = moves[i + 1]
        
        # Rounding Up
        # If the Neutral atom is moving from an AOD-steered tweezer to an SLM trap or vice versa
        # We approximate that has time to transfer_SLM_AOD + move from x1,y1 to x2,y2 using trapezoid.   
        q1_even = (x1 % 2 == 0) or (y1 % 2 == 0)
        q2_even = (x2 % 2 == 0) or (y2 % 2 == 0)
        if q1_even or q2_even:
            total_time += transfer_time
        dx = (x2 - x1)
        dy = (y2 - y1)

        # convert grid movement to physical distance
        d = (dx**2 + dy**2)**(0.5) * grid_spacing/2

        d_accel = v**2 / (2 * a)

        if d > 2 * d_accel:
            t = 2 * (v / a) + (d - 2 * d_accel) / v
        else:
            v_peak = (a * d)**(0.5)
            t = 2 * (v_peak / a)

        total_time += t

    return total_time
