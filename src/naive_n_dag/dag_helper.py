from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from fractions import Fraction
from pathlib import Path
import re

from qiskit import QuantumCircuit
from qiskit.circuit import Barrier
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode

from qiskit.circuit.library.standard_gates import CXGate, CZGate
from qiskit.circuit.library.standard_gates import SwapGate


def load_qasm_to_circuit(qasm_path: str | Path) -> QuantumCircuit:
    """Load a QASM file into a QuantumCircuit."""
    return QuantumCircuit.from_qasm_file(str(qasm_path))


def load_qasm_to_dag(qasm_path: str | Path) -> DAGCircuit:
    """Load a QASM file and convert it to a DAGCircuit."""
    circuit = load_qasm_to_circuit(qasm_path)
    return circuit_to_dag(circuit)


def _with_global_barriers(dag: DAGCircuit) -> DAGCircuit:
    """Return a DAG where every source barrier spans every quantum wire."""
    result = dag.copy_empty_like()
    for node in sorted(dag.op_nodes(), key=lambda item: item._node_id):
        if node.op.name == "barrier":
            result.apply_operation_back(Barrier(len(dag.qubits)), dag.qubits, [])
        else:
            result.apply_operation_back(node.op, node.qargs, node.cargs)
    return result


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


def format_gate_line(gate_name: str, gate_params: list, qubit_ids: list[int]) -> str:
    """Build a single OpenQASM-like gate line from normalized gate fields."""
    if gate_params:
        params_str = ", ".join(_format_gate_param(param) for param in gate_params)
        return f"{gate_name}({params_str}) " + ",".join(f"q[{qid}]" for qid in qubit_ids) + ";"
    return f"{gate_name} " + ",".join(f"q[{qid}]" for qid in qubit_ids) + ";"


def extract_index_from_bit(bit) -> int:
    """Extract the integer index from a Qiskit bit repr (e.g. Qubit(index=3))."""
    rep = repr(bit)
    if "index=" not in rep:
        raise ValueError(f"Unable to extract bit index from {rep}")
    idx_str = rep.split("index=", 1)[1].split(">", 1)[0].strip()
    return int(idx_str)


def format_classical_bit(bit) -> str:
    """Render a classical bit with its QASM register name and local index."""
    register = getattr(bit, "_register", None)
    index = getattr(bit, "_index", None)
    if register is not None and index is not None:
        return f"{register.name}[{int(index)}]"

    # Standalone bits do not have a register. Keep the historical default name.
    return f"c[{extract_index_from_bit(bit)}]"


def format_node_line(node: DAGOpNode) -> str:
    """Render a DAG op node as a schedule line, including measurement formatting."""
    qubit_indices = [extract_index_from_bit(q) for q in node.qargs]
    if node.op.name == "measure":
        if len(node.qargs) != 1 or len(node.cargs) != 1:
            raise ValueError("Measurement node must have exactly one qarg and one carg.")
        classical_bit = format_classical_bit(node.cargs[0])
        return f"measure q[{qubit_indices[0]}] -> {classical_bit};"

    params = [float(p) for p in getattr(node.op, "params", [])]
    return format_gate_line(node.op.name, params, qubit_indices)


def build_two_qubit_only_dag_with_single_qubit_context(
    dag: DAGCircuit,
) -> tuple[DAGCircuit, list[list[DAGOpNode]]]:
    """Return a 2Q-only DAG and non-2Q ops grouped by preceding 2Q layer boundary.

    The returned tuple is:
    - `two_qubit_dag`: DAG containing 2Q operations and barrier separators.
    - `single_layers`: list whose length is `len(list(two_qubit_dag.layers())) + 1`.
      `single_layers[i]` holds lines to run before two-qubit layer `i`, and the
      last entry holds lines to run after the final two-qubit layer.

    Measurements are included in `single_layers`. Barriers are retained in the
    reduced DAG as full-width separator layers and are not returned as physical
    operations. This makes every barrier a global scheduling boundary.
    """
    two_qubit_dag = dag.copy_empty_like()
    # For each qubit, track the next 2Q layer index after the latest 2Q op touching it.
    qubit_twoq_progress: dict[int, int] = {}
    num_twoq_layers = 0
    single_layers: list[list[DAGOpNode]] = [[]]

    for node in sorted(dag.op_nodes(), key=lambda item: item._node_id):
        if isinstance(node.op, SwapGate):
            continue

        if node.op.name == "barrier":
            boundary = max(qubit_twoq_progress.values(), default=num_twoq_layers)
            two_qubit_dag.apply_operation_back(
                Barrier(len(two_qubit_dag.qubits)), two_qubit_dag.qubits, []
            )
            next_layer = boundary + 1
            qubit_twoq_progress = {
                extract_index_from_bit(qubit): next_layer for qubit in dag.qubits
            }
            num_twoq_layers = max(num_twoq_layers, next_layer)
            while len(single_layers) <= next_layer:
                single_layers.append([])
            continue

        num_qargs = len(node.qargs)
        if num_qargs == 2:
            q0 = extract_index_from_bit(node.qargs[0])
            q1 = extract_index_from_bit(node.qargs[1])
            layer_idx = max(qubit_twoq_progress.get(q0, 0), qubit_twoq_progress.get(q1, 0))
            qubit_twoq_progress[q0] = layer_idx + 1
            qubit_twoq_progress[q1] = layer_idx + 1
            num_twoq_layers = max(num_twoq_layers, layer_idx + 1)
            two_qubit_dag.apply_operation_back(node.op, node.qargs, node.cargs)
            continue

        if num_qargs == 1:
            qid = extract_index_from_bit(node.qargs[0])
            bucket_idx = qubit_twoq_progress.get(qid, 0)
            while len(single_layers) <= bucket_idx:
                single_layers.append([])
            single_layers[bucket_idx].append(node)

    while len(single_layers) < num_twoq_layers + 1:
        single_layers.append([])

    return two_qubit_dag, single_layers


def _is_reorderable_cz(node: DAGOpNode) -> bool:
    return isinstance(node.op, CZGate) and getattr(node.op, "condition", None) is None


def _color_cz_block(nodes: list[DAGOpNode]) -> list[list[DAGOpNode]]:
    """Color a bipartite CZ multigraph optimally; use greedy coloring otherwise.

    Pad the bipartite graph to a balanced Delta-regular multigraph, then remove
    a perfect matching for each color. Dummy edges are discarded. Repeated
    gates remain distinct edges; no cancellation or gate synthesis is done.
    """
    adjacency = defaultdict(set)
    for node in nodes:
        a, b = node.qargs
        adjacency[a].add(b)
        adjacency[b].add(a)
    order = {q: i for i, q in enumerate(adjacency)}
    side = {}
    bipartite = True
    for root in adjacency:
        if root in side:
            continue
        side[root] = 0
        queue = deque([root])
        while queue:
            a = queue.popleft()
            # Traverse in first-appearance order for reproducible matchings.
            for b in sorted(adjacency[a], key=lambda q: order[q]):
                if b not in side:
                    side[b] = 1 - side[a]
                    queue.append(b)
                elif side[b] == side[a]:
                    bipartite = False
    if not bipartite:
        layers, occupied = [], []
        for node in nodes:
            operands = set(node.qargs)
            for layer, used in zip(layers, occupied):
                if not operands & used:
                    layer.append(node)
                    used.update(operands)
                    break
            else:
                layers.append([node])
                occupied.append(operands)
        return layers

    left = {q: i for i, q in enumerate(q for q in adjacency if side[q] == 0)}
    right = {q: i for i, q in enumerate(q for q in adjacency if side[q] == 1)}
    size = max(len(left), len(right))
    edges = [defaultdict(deque) for _ in range(size)]
    degree_left, degree_right = [0] * size, [0] * size
    for node in nodes:
        a, b = node.qargs
        if side[a] == 1:
            a, b = b, a
        u, v = left[a], right[b]
        edges[u][v].append(node)
        degree_left[u] += 1
        degree_right[v] += 1
    delta = max(degree_left + degree_right, default=0)
    v = 0
    for u in range(size):
        while degree_left[u] < delta:
            while degree_right[v] == delta:
                v += 1
            count = min(delta - degree_left[u], delta - degree_right[v])
            edges[u][v].extend([None] * count)
            degree_left[u] += count
            degree_right[v] += count

    layers = []
    for _ in range(delta):
        match_left, match_right = {}, {}
        for start in range(size):
            # Augment iteratively to avoid recursion limits on larger codes.
            queue = deque([start])
            previous = {}
            seen_left = {start}
            end = None
            while queue and end is None:
                u = queue.popleft()
                for v in sorted(edges[u]):
                    if not edges[u][v] or v in previous:
                        continue
                    previous[v] = u
                    if v not in match_right:
                        end = v
                        break
                    nxt = match_right[v]
                    if nxt not in seen_left:
                        seen_left.add(nxt)
                        queue.append(nxt)
            if end is None:
                raise ValueError("Regular bipartite CZ graph has no perfect matching.")
            while end is not None:
                u = previous[end]
                old = match_left.get(u)
                match_left[u], match_right[end] = end, u
                end = old
        layer = []
        for u, v in sorted(match_left.items()):
            node = edges[u][v].popleft()
            if node is not None:
                layer.append(node)
        if layer:
            layers.append(layer)
    return layers


def _validate_cz_reordering(original, reordered):
    """Check occurrence preservation and wire order modulo adjacent CZ swaps.

    Occurrence IDs distinguish identical gates. For every quantum/classical
    wire, only consecutive runs of unconditional CZs may be permuted. Thus
    H, reset, measurement, barriers, parameters and destinations stay intact.
    """
    if Counter(n._node_id for n in original) != Counter(n._node_id for n in reordered):
        raise ValueError("CZ preprocessing changed operation occurrences.")

    def traces(nodes):
        wires = defaultdict(list)
        for node in nodes:
            for wire in (*node.qargs, *node.cargs):
                wires[wire].append(node)
        result = {}
        for wire, sequence in wires.items():
            trace, run = [], []
            for node in sequence:
                if _is_reorderable_cz(node):
                    run.append(node._node_id)
                else:
                    trace.extend((tuple(sorted(run)), node._node_id))
                    run = []
            trace.append(tuple(sorted(run)))
            result[wire] = trace
        return result

    if traces(original) != traces(reordered):
        raise ValueError("CZ preprocessing crossed a noncommuting operation.")


def _commuting_cz_sections(reference_nodes):
    """Yield non-CZ operations and reachable CZ blocks without crossing barriers."""
    original = list(reference_nodes)
    if any(getattr(n.op, "condition", None) is not None or
           getattr(n.op, "blocks", ()) for n in original):
        raise ValueError("CZ preprocessing does not support conditional gates or control flow.")
    all_wires = {wire for node in original for wire in (*node.qargs, *node.cargs)}
    last = {}
    successors = defaultdict(set)
    pending = {}
    for index, node in enumerate(original):
        wires = all_wires if node.op.name == "barrier" else (*node.qargs, *node.cargs)
        predecessors = {last[wire] for wire in wires if wire in last}
        pending[index] = len(predecessors)
        for predecessor in predecessors:
            successors[predecessor].add(index)
        for wire in wires:
            last[wire] = index
    ready = {index for index, count in pending.items() if count == 0}

    def consume(index):
        ready.remove(index)
        for successor in successors[index]:
            pending[successor] -= 1
            if pending[successor] == 0:
                ready.add(successor)

    while ready:
        while True:
            non_cz = sorted(index for index in ready if not _is_reorderable_cz(original[index]))
            if not non_cz:
                break
            for index in non_cz:
                yield False, [original[index]]
                consume(index)
        block = []
        while True:
            cz = sorted(index for index in ready if _is_reorderable_cz(original[index]))
            if not cz:
                break
            for index in cz:
                block.append(original[index])
                consume(index)
        if block:
            yield True, block


def reorder_cz_layers_by_displacement(reference_nodes, homes, config):
    """Re-layer commuting CZs after placement; do not group loads or plan motion.

    Return explicit disjoint 2Q layers, their 1Q contexts, and the checked
    reference order. Keep these together: ASAP re-extraction could split the
    equal-displacement bundles. Only geometry and ancilla preference are used;
    Held-Karp, side selection, grouping, reuse and shuttling belong to dynamics.
    """
    if reference_nodes is None:
        raise ValueError("Displacement reordering requires full circuit context.")
    original = list(reference_nodes)
    ancillas = {q for node in original if node.op.name == "reset"
                for q in op_node_signature(node)[2]}
    layers, singles, reordered = [], [[]], []

    def append_layer(layer):
        layers.append(layer)
        singles.append([])
        reordered.extend(layer)

    for is_cz, nodes in _commuting_cz_sections(original):
        if is_cz:
            buckets = defaultdict(list)
            for node in nodes:
                pair = op_node_signature(node)[2]
                preferred = [q for q in pair if q in ancillas]
                mover = (preferred[0] if config.get("prefer_moving_ancillas", False)
                         and len(preferred) == 1 else pair[0])
                partner = pair[1] if mover == pair[0] else pair[0]
                displacement = tuple(homes[partner][axis] - homes[mover][axis] for axis in (0, 1))
                buckets[displacement].append(node)
            # Keep each equal-displacement matching intact. Compatible
            # matchings can share a layer so dynamics can group their movers
            # and choose the travel order; no travel cost is evaluated here.
            bundles = [matching for bucket in buckets.values()
                       for matching in _color_cz_block(bucket)]
            bundles.sort(key=lambda matching: -len(matching))
            packed = []
            for matching in bundles:
                operands = {q for node in matching for q in node.qargs}
                for layer, used in packed:
                    if not operands & used:
                        layer.extend(matching)
                        used.update(operands)
                        break
                else:
                    packed.append((list(matching), set(operands)))
            for layer, _ in packed:
                append_layer(layer)
        else:
            node = nodes[0]
            if node.op.name == "barrier" or len(node.qargs) == 1:
                # Separate repeated classical writes: the 1Q pulse packer
                # tracks quantum operands, not classical destinations.
                if node.cargs and any(set(node.cargs) & set(previous.cargs) for previous in singles[-1]):
                    append_layer([])
                singles[-1].append(node)
                reordered.append(node)
            elif len(node.qargs) == 2 and node.op.name != "swap":
                append_layer([node])
            else:
                raise ValueError("Displacement reordering requires a SWAP-filtered 1Q/2Q circuit.")
    _validate_cz_reordering(original, reordered)
    return layers, singles, reordered


def reorder_commuting_cz_blocks(dag: DAGCircuit) -> DAGCircuit:
    """Rebuild a full DAG with colored commuting CZ blocks.

    Drain ready non-CZ operations, then collect all CZs reachable without
    executing another non-CZ operation. Color that block and repeat. This is
    conservative across block boundaries and requires no BB-specific labels.
    """
    dag = _with_global_barriers(dag)
    original = list(dag.topological_op_nodes())
    if any(getattr(n.op, "condition", None) is not None or
           getattr(n.op, "blocks", ()) for n in original):
        raise ValueError("CZ preprocessing does not support conditional gates or control flow.")
    reordered = []
    for is_cz, nodes in _commuting_cz_sections(original):
        if is_cz:
            for layer in _color_cz_block(nodes):
                reordered.extend(layer)
        else:
            reordered.extend(nodes)
    _validate_cz_reordering(original, reordered)
    result = dag.copy_empty_like()
    for node in reordered:
        result.apply_operation_back(node.op, node.qargs, node.cargs)
    return result


def load_qasm_to_two_qubit_dag_with_single_qubit_context(
    qasm_path: str | Path, *, reorder_cz: bool = False,
    return_reference_nodes: bool = False,
):
    """Prepare 2Q DAG/context, optionally reordering CZs before extraction.

    Default return remains ``(two_qubit_dag, single_layers)``. With
    ``return_reference_nodes=True``, a third item contains the full prepared
    SWAP-filtered reference for strict schedule validation. Color order becomes
    wire order; subsequent DAG layering may compact independent color groups.
    Keep the original full DAG and context if the proposed ordering increases
    extracted two-qubit depth. Equal-depth proposals are accepted.
    """
    dag = _with_global_barriers(load_qasm_to_dag(qasm_path))
    result = build_two_qubit_only_dag_with_single_qubit_context(dag)
    if reorder_cz:
        proposed_dag = reorder_commuting_cz_blocks(dag)
        proposed_result = build_two_qubit_only_dag_with_single_qubit_context(proposed_dag)
        original_depth = sum(
            any(node.op.name != "barrier" for node in layer["graph"].op_nodes())
            for layer in result[0].layers()
        )
        proposed_depth = sum(
            any(node.op.name != "barrier" for node in layer["graph"].op_nodes())
            for layer in proposed_result[0].layers()
        )
        if proposed_depth <= original_depth:
            dag, result = proposed_dag, proposed_result
    if return_reference_nodes:
        from naive_dag.dag_helper import dag_with_gate_ops_only
        return (*result, list(dag_with_gate_ops_only(dag)))
    return result


_TXT_GATE_LINE_RE = re.compile(r"^\s*(\d+)\s*,\s*(\d+)\s*:\s*(\d+)\s+(\d+)\s*$")
_TXT_SUMMARY_LINE_RE = re.compile(r"^\s*T\s*=\s*(\d+)\s*,\s*cx\s*=\s*(\d+)\s*$")


def dag_from_txt_auto(txt_path: str | Path) -> tuple[DAGCircuit, list[list[DAGOpNode]]]:
    """Build a two-qubit DAG from a timestep text file.

    Expected format per timestep:
    - one or more gate lines: ``<T>, <gate_index>: <q0> <q1>``
    - followed by summary line: ``T = <T>, cx = <count>``

    Example:
    ``4, 4: 1 2``
    ``4, 6: 0 3``
    ``T = 4, cx = 2``

    Returns:
    - `two_qubit_dag`: a DAG containing only CX operations from the text file.
    - `single_layers`: empty single-qubit context buckets with length
      `len(list(two_qubit_dag.layers())) + 1`.
    """
    path = Path(txt_path)
    if not path.exists():
        raise FileNotFoundError(f"txt gate-order file not found: {path}")

    steps: list[list[tuple[int, int]]] = []
    current_timestep: int | None = None
    current_pairs: list[tuple[int, int]] = []
    expected_next_timestep: int | None = None
    max_qubit_id = -1
    saw_content = False

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            saw_content = True

            gate_match = _TXT_GATE_LINE_RE.match(line)
            if gate_match is not None:
                timestep, _gate_index, q0, q1 = (int(v) for v in gate_match.groups())
                if q0 == q1:
                    raise ValueError(
                        f"{path}:{line_no}: invalid two-qubit gate uses the same qubit id twice: {q0}."
                    )

                if current_timestep is None:
                    if expected_next_timestep is not None and timestep != expected_next_timestep:
                        raise ValueError(
                            f"{path}:{line_no}: expected timestep {expected_next_timestep}, got {timestep}."
                        )
                    current_timestep = timestep
                    current_pairs = []
                elif timestep != current_timestep:
                    raise ValueError(
                        f"{path}:{line_no}: encountered timestep {timestep} before summary line for "
                        f"timestep {current_timestep}."
                    )

                current_pairs.append((q0, q1))
                max_qubit_id = max(max_qubit_id, q0, q1)
                continue

            summary_match = _TXT_SUMMARY_LINE_RE.match(line)
            if summary_match is not None:
                summary_timestep, cx_count = (int(v) for v in summary_match.groups())
                if current_timestep is None:
                    raise ValueError(
                        f"{path}:{line_no}: found summary line before any gate lines for timestep {summary_timestep}."
                    )
                if summary_timestep != current_timestep:
                    raise ValueError(
                        f"{path}:{line_no}: summary timestep {summary_timestep} does not match "
                        f"gate timestep {current_timestep}."
                    )
                if cx_count != len(current_pairs):
                    raise ValueError(
                        f"{path}:{line_no}: summary cx={cx_count} does not match {len(current_pairs)} gate "
                        f"line(s) for timestep {current_timestep}."
                    )

                steps.append(current_pairs)
                expected_next_timestep = current_timestep + 1
                current_timestep = None
                current_pairs = []
                continue

            raise ValueError(
                f"{path}:{line_no}: invalid format. Expected '<T>, <gate_index>: <q0> <q1>' or "
                "'T = <T>, cx = <count>'."
            )

    if not saw_content:
        raise ValueError(f"{path}: file is empty; expected timestep gate lines.")
    if current_timestep is not None:
        raise ValueError(
            f"{path}: missing summary line for final timestep {current_timestep}."
        )
    if not steps:
        raise ValueError(f"{path}: no timestep blocks parsed from file.")

    # Build an empty circuit only to allocate the required qubit register.
    circuit = QuantumCircuit(max_qubit_id + 1)
    two_qubit_dag = circuit_to_dag(circuit)
    qubits = two_qubit_dag.qubits
    for timestep_pairs in steps:
        for q0, q1 in timestep_pairs:
            two_qubit_dag.apply_operation_back(CXGate(), qargs=[qubits[q0], qubits[q1]], cargs=[])

    single_layers: list[list[DAGOpNode]] = [[] for _ in range(len(list(two_qubit_dag.layers())) + 1)]
    return two_qubit_dag, single_layers
