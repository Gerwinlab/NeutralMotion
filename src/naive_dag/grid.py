from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
from pint import Quantity

@dataclass
class GridNode:
    x: Quantity
    y: Quantity
    row: int
    col: int
    qubit: Optional[int] = None
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
    x: Quantity
    y: Quantity
    row: int
    col: int
    qubit: Optional[int] = None

    def is_occupied(self)->bool:
        """Return True when this lattice site currently has an assigned qubit."""
        return self.qubit is not None
@dataclass
class Qubit:
    """
    Represents a logical qubit located at a node.
    """
    id: int
    node: GridNode

    def position(self)->Tuple[float,float]:
        """Return the physical (x, y) position for this qubit."""
        return self.node.x, self.node.y
    def grid_position(self)->Tuple[int,int]:
        """Return integer grid coordinates ``(row, col)`` for this qubit."""
        return self.node.row,self.node.col
    
def generate_grid(dimensions: List[int],rydberg_radius) -> List[List[GridNode]]:
    """Construct a 2D lattice where adjacent logical trap points are 2 indices apart."""
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
    """Create a qubit id and place it at a specific grid node."""
    node = grid[row][col]
    if node.is_occupied():
        raise ValueError(f"Node ({row},{col}) is already occupied.")
    qubit = Qubit(id=qubit_id,node=node)
    node.qubit = qubit
    return qubit

def move_qubit(qubit: Qubit, new_node: GridNode) -> None:
    """Move ``qubit`` to ``new_node`` while preserving occupancy invariants."""
    # If the node is occupied by a *different* qubit, error
    if new_node.is_occupied() and new_node.qubit is not qubit:
        raise ValueError("Target node is already occupied by another qubit.")

    # If it's the same node, nothing to do
    if new_node is qubit.node:
        return

    # Free the old node
    qubit.node.qubit = None

    # Occupy the new node
    new_node.qubit = qubit
    qubit.node = new_node


def naive_fill(grid: List[List[GridNode]], n:int, seed: int=0, random_fill: bool = True) -> List[Qubit]:
    """Place ``n`` qubits on available even-even nodes, randomly or row-major."""

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
