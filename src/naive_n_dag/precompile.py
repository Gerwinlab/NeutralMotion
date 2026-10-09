"""Placement-aware, dependency-safe CZ pulse packing before physical routing."""
from __future__ import annotations

from collections import defaultdict

from .dag_helper import _is_reorderable_cz, _validate_cz_reordering, op_node_signature
from .dynamics import _preferred_mover


def _pack_displacement_block(nodes, homes, config):
    """Greedily pack whole equal-displacement matchings into disjoint layers.

    Keeping a matching intact avoids splitting a potential shared pulse just
    to reduce depth. Conflicting occurrences (including repeated gates) still
    require separate matchings. AOD span/alignment and actual pulse legality
    remain the responsibility of grouping and shuttling.
    """
    buckets = defaultdict(list)
    for node in nodes:
        pair = op_node_signature(node)[2]
        mover = _preferred_mover(pair, config)
        if mover is None:
            mover = pair[0]  # Same default as a fresh dynamics group.
        partner = pair[1] if mover == pair[0] else pair[0]
        vector = tuple(homes[partner][axis] - homes[mover][axis] for axis in (0, 1))
        buckets[vector].append(node)

    bundles = []
    for vector, bucket in buckets.items():
        matchings = []
        for node in bucket:
            operands = set(node.qargs)
            for matching, used in matchings:
                if not operands & used:
                    matching.append(node)
                    used.update(operands)
                    break
            else:
                matchings.append(([node], set(operands)))
        bundles.extend((vector, matching, used) for matching, used in matchings)

    # Largest bundles first is a deterministic bin-packing heuristic, not an
    # optimal edge coloring. Never break a shared pulse to fill a partial layer.
    bundles.sort(key=lambda bundle: (-len(bundle[1]), sum(v * v for v in bundle[0]), bundle[0]))
    layers = []
    for vector, matching, operands in bundles:
        for layer, used, vectors in layers:
            if not operands & used:
                layer.extend(matching)
                used.update(operands)
                vectors.append(vector)
                break
        else:
            layers.append((list(matching), set(operands), [vector]))

    # Start with a layer having a near-home target. All layers in this block
    # commute; routing within each layer is still handled by Held-Karp/alignment.
    layers.sort(key=lambda layer: min(sum(v * v for v in vector) for vector in layer[2]))
    return [layer for layer, _, _ in layers]


def precompile_displacement_layers(reference_nodes, homes, config):
    """Return explicit (2Q layers, single contexts, reordered reference).

    Only unconditional CZs commute here. Drain ready non-CZ operations, then
    collect CZs reachable without crossing another non-CZ operation. Quantum
    and classical wire dependencies, and global barrier epochs, are retained.
    Explicit layers prevent a subsequent DAG ASAP compaction from splitting
    the displacement bundles apart again. Inputs and homes are not mutated.
    """
    return _precompile_cz_context(reference_nodes, homes, config, _pack_displacement_block)


def precompile_commuting_blocks(reference_nodes, homes, config):
    """Retain whole commuting CZ blocks instead of imposing disjoint layers.

    Blocks can contain overlapping operands. They are inputs to phase routing,
    never to the ordinary disjoint-layer scheduler. Non-CZ 2Q gates remain
    singleton blocks, with the same single-qubit contexts and barriers.
    """
    return _precompile_cz_context(
        reference_nodes, homes, config, lambda nodes, homes, config: [nodes] if nodes else [],
    )


def _precompile_cz_context(reference_nodes, homes, config, pack_block):
    original = list(reference_nodes)
    if any(getattr(n.op, "condition", None) is not None or
           getattr(n.op, "blocks", ()) for n in original):
        raise ValueError("Displacement preprocessing does not support conditional gates or control flow.")
    config = dict(config, _ancilla_qubits=frozenset(
        q for node in original if node.op.name == "reset"
        for q in op_node_signature(node)[2]
    ))
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

    layers, singles, reordered = [], [[]], []

    def append_layer(layer):
        layers.append(layer)
        singles.append([])
        reordered.extend(layer)

    while ready:
        while True:
            non_cz = sorted(index for index in ready if not _is_reorderable_cz(original[index]))
            if not non_cz:
                break
            for index in non_cz:
                node = original[index]
                if node.op.name == "barrier" or len(node.qargs) == 1:
                    # The 1Q pulse packer tracks quantum wires, not classical
                    # destinations. Keep repeated classical writes in separate
                    # context buckets so packing cannot make them simultaneous.
                    if node.cargs and any(set(node.cargs) & set(previous.cargs) for previous in singles[-1]):
                        append_layer([])
                    singles[-1].append(node)
                    reordered.append(node)
                elif len(node.qargs) == 2 and node.op.name != "swap":
                    append_layer([node])
                else:
                    raise ValueError("Displacement preprocessing requires a SWAP-filtered 1Q/2Q circuit.")
                consume(index)
        block = []
        while True:
            cz = sorted(index for index in ready if _is_reorderable_cz(original[index]))
            if not cz:
                break
            for index in cz:
                block.append(original[index])
                consume(index)
        for layer in pack_block(block, homes, config):
            append_layer(layer)

    _validate_cz_reordering(original, reordered)
    return layers, singles, reordered
