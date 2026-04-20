#TODO: Check if the number of traps is enough for the number of qubits and warn if your is an excess number of traps

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple
import random
"""
grid.py

Utilities for generating and managing a neutral atom grid with spacing defined by a rydberg radius.

Each grid node stores:
    -Physical coordinates (x,y), starting at (0,0)
    -Grid indices (row,col)
    -qubit assignment
"""

@dataclass
class GridNode:
    """
    Represents a single lattice site in the neutral atom array.
    """
    x: float
    y: float
    row: int
    col: int
    qubit: Optional[int] = None

    def is_occupied(self)->bool:
        """Returns true if a qubit is assigned"""
        return self.qubit is not None
@dataclass
class Qubit:
    """
    Represents a logical qubit located at a node.
    """
    id: int
    node: GridNode

    def position(self)->Tuple[float,float]:
        """Returns (x,y) position"""
        return self.node.x, self.node.y
    def grid_position(self)->Tuple[int,int]:
        """Returns row and col"""
        return self.node.row,self.node.col
    
def generate_grid(dimensions: List[int],rydberg_radius:float) -> List[List[GridNode]]:
    """
    Construct a 2D grid of Nodes spaced according to the given Rydberg radius.

    Parameters
    ----------
    dimension : List[int] - given by initializing json file.
    rydberg_radius : float - given by initializing json file.

    Returns
    -------
    List[List[GridNode]] - Matrix of Nodes that contain qubits.
    """
    rows,cols=dimensions
    grid = []

    for row in range(2*rows-1):
        row_nodes = []
        for col in range(2*cols-1):
            x = col * rydberg_radius/2
            y = row * rydberg_radius/2
            node = GridNode(x=x,y=y,row=row,col=col,qubit=None)
            row_nodes.append(node)
        grid.append(row_nodes)
    return grid

def place_qubit(grid: List[List[GridNode]],row:int, col:int,qubit_id: int) ->Qubit:
    """
    Create and place a qubit at a grid location

    Parameters
    ----------
    Matrix/Grid : List[List[GridNode]] - created by generate_grid and has dimensions specified in Json file
    row : Int
    col : Int - location of node in grid
    qubit_id : int - number of qubit that is placed in spot.
    """
    node = grid[row][col]
    if node.is_occupied():
        raise ValueError(f"Node ({row},{col}) is already occupied.")
    qubit = Qubit(id=qubit_id,node=node)
    node.qubit = qubit
    return qubit

def move_qubit(qubit: Qubit, new_node: GridNode) -> None:
    """
    Place a qubit in a different grid location
    """
    if new_node.is_occupied():
        raise ValueError("Target node is already occupied.")
    qubit.node.qubit = None

    new_node.qubit = qubit
    qubit.node = new_node

def naive_fill(grid: List[List[GridNode]], n:int, seed: int=0, random_fill: bool = True) -> List[Qubit]:
    """
    Assign qubits 0..n-1 to grid locations where both row and column indices are even. Qubits should be placed on the outside and have a minor node between them.

    Parameters
    ----------
    Grid : 2D list of GridNoe
    n : number of qubits
    seed : random seed (used only if random_fill=True)
    random_fill : bool
        If True, randomly assign positions.
        If False, fill in increasing row-major order.

    Returns
    -------
    List[Qubit]
        List of placed qubit objects
    """

    valid_nodes = [
        node
        for row in grid
        for node in row
        if node.row % 2 == 0 and node.col % 2 == 0 and not node.is_occupied()
    ]

    if n > len(valid_nodes):
        raise ValueError(f"Not enough valid even-even positions for {n} qubits.")
    
    if random_fill:
        random.seed(seed)
        selected_nodes=random.sample(valid_nodes,n)
    else:
        selected_nodes = valid_nodes[:n]
    qubits = []
    for qubit_id, node in enumerate(selected_nodes):
        qubit = Qubit(id=qubit_id,node=node)
        node.qubit = qubit
        qubits.append(qubit)
    return qubits


def Fastsa_Fill(
    grid: List[List[GridNode]],
    n: int,
    two_qubit_dag,
    config: Mapping[str, Any],
    *,
    seed: int = 0,
    stage1_iterations: int = 10,
    stage2_iterations: int = 100,
    initial_accept_prob: float = 0.99,
    c: float = 100.0,
    stage3_temperature_threshold: float = 1e-6,
    stage3_max_iterations: int = 1000,
    stage3_section_size: int = 4
) -> List[Qubit]:
    """Place qubits using Fast-SA best positions and return placed qubit objects."""
    from .FastSA_Solver import run_fastsa
    result = run_fastsa(
        grid,
        n,
        two_qubit_dag,
        config,
        stage1_iterations=stage1_iterations,
        stage2_iterations=stage2_iterations,
        initial_accept_prob=initial_accept_prob,
        c=c,
        stage3_temperature_threshold=stage3_temperature_threshold,
        stage3_max_iterations=stage3_max_iterations,
        stage3_section_size=stage3_section_size,
        seed=seed,
    )

    # Clear any existing occupancy before placing the optimized configuration.
    for row_nodes in grid:
        for node in row_nodes:
            node.qubit = None

    qubits: List[Qubit] = []
    for qubit_id in range(n):
        if qubit_id not in result.best_positions:
            raise ValueError(f"Fast-SA result missing position for qubit id {qubit_id}.")
        col, row = result.best_positions[qubit_id]
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
            raise ValueError(f"Fast-SA proposed out-of-bounds position for q[{qubit_id}]: ({row},{col})")
        qubits.append(place_qubit(grid, row, col, qubit_id))
    return qubits
