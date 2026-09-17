"""Build a qSIEVE-inspired schedule without changing input wire order.

Run with ../Phys765_Test/bin/python bb144_paper_schedule.py.
The optimizer visits equal check/data displacement groups with an exact
precedence-constrained dynamic program, choosing among four legal gate sides.
It uses the existing naive_n_dag geometry, timer, validator, and text writer.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from naive_n_dag.dag_helper import load_qasm_to_dag, format_node_line, op_node_signature
from naive_n_dag.dynamics import (
    _CARDINAL, _event_batches, _gate_parts, _motion_parameters, _step_seconds,
    parity_route_moves, schedule_duration, validate_schedule,
)
from naive_n_dag.grid import generate_grid, initial_layout_fill
from naive_n_dag.main import _coerce_unit_config
from naive_n_dag.scheduling import single_qubit_layer_time, write_timed_schedule


def groups_and_precedence(nodes, homes, check_ids, split_check_types=False):
    groups = {}
    previous = {}
    edges = set()
    for node in nodes:
        ids = op_node_signature(node)[2]
        check, = set(ids) & set(check_ids)
        data, = set(ids) - {check}
        offset = tuple(homes[data][k] - homes[check][k] for k in range(2))
        key = (check // 72, offset) if split_check_types else offset
        if key not in groups:
            groups[key] = []
        groups[key].append(node)
        for q in ids:
            if q in previous and previous[q] != key:
                edges.add((previous[q], key))
            previous[q] = key
    keys = list(groups)
    predecessors = [0] * len(keys)
    for a, b in edges:
        predecessors[keys.index(b)] |= 1 << keys.index(a)
    offsets = [key[1] for key in keys] if split_check_types else keys
    return offsets, list(groups.values()), predecessors


def optimize_tour(offsets, predecessors, config):
    """Exact optimum within a single rigid load and one visit per group."""
    sides = [tuple(v[k] + d[k] for k in range(2)) for v in offsets for d in _CARDINAL]
    positions = sides + list(_CARDINAL)
    n = len(offsets)
    full = (1 << n) - 1

    @lru_cache(None)
    def route(a, b):
        return tuple(parity_route_moves(positions[a], positions[b], config))

    costs = [[sum(_step_seconds(d, config) for d in route(a, b))
              for b in range(len(positions))] for a in range(len(positions))]
    transfer_motion = _step_seconds((1, 0), config)
    decisions = {}

    @lru_cache(None)
    def solve(mask, last):
        if mask == full:
            return min(costs[last][4*n+s] + transfer_motion for s in range(4))
        best = float("inf")
        choice = None
        for j in range(n):
            if mask >> j & 1 or predecessors[j] & mask != predecessors[j]:
                continue
            for side in range(4):
                nxt = 4*j + side
                value = costs[last][nxt] + solve(mask | 1 << j, nxt)
                if value < best:
                    best, choice = value, (j, nxt)
        if choice is not None:
            decisions[mask, last] = choice
        return best

    start = min(range(4*n, 4*n+4), key=lambda i: solve(0, i))
    if solve(0, start) == float("inf"):
        raise ValueError("Grouping induced a precedence cycle; cannot reorder these groups.")
    steps = [(positions[start], None)]
    visits = []
    mask, last = 0, start
    while mask != full:
        j, nxt = decisions[mask, last]
        steps.extend((d, None) for d in route(last, nxt))
        steps.append(((0, 0), j))
        visits.append({"group": j, "offset": offsets[j], "aod_displacement": positions[nxt]})
        mask |= 1 << j
        last = nxt
    end = min(range(4*n, 4*n+4), key=lambda i: costs[last][i])
    steps.extend((d, None) for d in route(last, end))
    steps.append((tuple(-v for v in positions[end]), None))
    return steps, visits, transfer_motion + solve(0, start)


def emit_phase(nodes, check_ids, homes, config):
    offsets, groups, predecessors = groups_and_precedence(nodes, homes, check_ids)
    steps, visits, optimal_motion = optimize_tour(offsets, predecessors, config)
    events = []
    displacement = (0, 0)
    for step, group in steps:
        if group is not None:
            events.append(("gate", " ".join(format_node_line(n) for n in groups[group])))
        else:
            for q in check_ids:
                start = tuple(homes[q][k] + displacement[k] for k in range(2))
                end = tuple(start[k] + step[k] for k in range(2))
                events.append(("move", q, start, end))
            displacement = tuple(displacement[k] + step[k] for k in range(2))
    assert displacement == (0, 0)
    return events, {"check_ids": list(check_ids), "visits": visits,
                    "motion_us": optimal_motion * 1e6,
                    "group_sizes": [len(g) for g in groups],
                    "predecessor_masks": predecessors}


def emit_combined(nodes, homes, config):
    """Hold both check subarrays in one load; move past data in the SLM."""
    check_ids = range(144, 288)
    cz = [n for n in nodes if n.op.name == "cz"]
    offsets, groups, predecessors = groups_and_precedence(cz, homes, check_ids, True)
    steps, visits, optimal_motion = optimize_tour(offsets, predecessors, config)
    events, done = [], set()
    wire_nodes = defaultdict(list)
    for node in nodes:
        for q in op_node_signature(node)[2]:
            wire_nodes[q].append(node)
    cursor = {q: 0 for q in wire_nodes}

    def mark_done(node):
        for q in op_node_signature(node)[2]:
            assert wire_nodes[q][cursor[q]] is node, "Input wire order changed"
            cursor[q] += 1
        done.add(id(node))

    def flush_singles():
        ready = []
        for node in nodes:
            if id(node) in done or node.op.name == "cz":
                continue
            q, = op_node_signature(node)[2]
            if wire_nodes[q][cursor[q]] is node:
                ready.append(node)
                mark_done(node)
        lines, _ = single_qubit_layer_time(ready, config["average_single_gate_time"], config["t_switch"])
        events.extend(("gate", line) for line in lines)

    flush_singles()
    displacement = (0, 0)
    for step, group in steps:
        if group is not None:
            flush_singles()
            for node in groups[group]:
                mark_done(node)
            events.append(("gate", " ".join(format_node_line(n) for n in groups[group])))
        else:
            for q in check_ids:
                start = tuple(homes[q][k] + displacement[k] for k in range(2))
                end = tuple(start[k] + step[k] for k in range(2))
                events.append(("move", q, start, end))
            displacement = tuple(displacement[k] + step[k] for k in range(2))
    flush_singles()
    assert len(done) == len(nodes) and displacement == (0, 0)
    return events, {"check_ids": list(check_ids), "visits": visits,
                    "motion_us": optimal_motion * 1e6,
                    "group_sizes": [len(g) for g in groups], "predecessor_masks": predecessors}


def parse_serialized(path):
    """Read the written artifact independently of its in-memory events."""
    move_re = re.compile(r"(load|move|unload) q\[(\d+)\](?: ->)? "
                         r"\((-?\d+),(-?\d+)\) : \((-?\d+),(-?\d+)\)")
    events, homes, lines = [], {}, []
    expected_t = 0
    for line in path.read_text().splitlines():
        if line.startswith("T="):
            assert int(line[2:]) == expected_t
            expected_t += 1
        elif line.startswith("initialize "):
            entries = re.findall(r"q\[(\d+)\] -> \((-?\d+),(-?\d+)\)", line)
            homes = {int(q): (int(r), int(c)) for q, r, c in entries}
            assert len(homes) == len(entries)
        elif line.startswith(("load ", "move ", "unload ")):
            matches = list(move_re.finditer(line))
            assert matches and not move_re.sub("", line).strip()
            batch = []
            for match in matches:
                kind, *numbers = match.groups()
                q, r, c, rr, cc = map(int, numbers)
                expected = "load" if r % 2 == c % 2 == 0 else (
                    "unload" if rr % 2 == cc % 2 == 0 else "move")
                assert kind == expected
                batch.append(("move", q, (r, c), (rr, cc)))
            events.extend(batch)
            lines.append(batch)
        elif "q[" in line:
            events.append(("gate", line))
            lines.append([events[-1]])
    assert list(_event_batches(events)) == lines, "Serialization changed batch boundaries"
    return homes, events


def timing_breakdown(events, config):
    motion = transfer = pulses = 0.0
    counts = Counter()
    paper_motion = 0.0
    for batch in _event_batches(events):
        e = batch[0]
        if e[0] == "gate":
            kinds = {(p, len(ids)) for _, p, ids in _gate_parts(e[1])}
            for _, arity in kinds:
                duration = config["average_single_gate_time" if arity == 1 else "average_two_gate_time"]
                pulses += (duration + config["t_switch"]).to("microsecond").magnitude
                counts[f"{arity}q_pulses"] += 1
        else:
            step = tuple(e[3][k] - e[2][k] for k in range(2))
            motion += _step_seconds(step, config) * 1e6
            spacing = config["rydberg_radius"].to("micrometer").magnitude / 2
            paper_motion += sum((6 * abs(d) * spacing / .02) ** .5 for d in step)
            is_transfer = any(p[0] % 2 == p[1] % 2 == 0 for p in e[2:])
            if is_transfer:
                transfer += config["transfer_SLM_AOD"].to("microsecond").magnitude
            counts["transfer_batches" if is_transfer else "move_batches"] += 1
    return {"motion_us": motion, "transfer_us": transfer, "gate_us": pulses,
            "total_us": motion + transfer + pulses,
            "paper_eq3_same_segments_motion_us": paper_motion,
            "counts": dict(counts)}


def write_timeline(path, events, config, homes):
    """A compact, reviewable companion with explicit cumulative time."""
    current, solver_time, paper_time = dict(homes), 0.0, 0.0
    with path.open("w", newline="") as f:
        fields = ["T", "action", "atoms_or_gates", "delta_row", "delta_col",
                  "aod_offset_row", "aod_offset_col", "solver_duration_us", "solver_end_us",
                  "paper_eq3_motion_us", "paper_eq3_plus_solver_overheads_end_us"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t, batch in enumerate(_event_batches(events), 1):
            e = batch[0]
            detail = timing_breakdown(batch, config)
            solver_time += detail["total_us"]
            paper_time += detail["paper_eq3_same_segments_motion_us"] + detail["transfer_us"] + detail["gate_us"]
            if e[0] == "move":
                action = "load" if all(v % 2 == 0 for v in e[2]) else (
                    "unload" if all(v % 2 == 0 for v in e[3]) else "move")
                step = tuple(e[3][k] - e[2][k] for k in range(2))
                for _, q, _, end in batch:
                    current[q] = end
                count = len(batch)
            else:
                action, step = "gate", (0, 0)
                count = len(list(_gate_parts(e[1])))
            loaded = [q for q, pos in current.items() if any(v % 2 for v in pos)]
            offset = tuple(current[loaded[0]][k] - homes[loaded[0]][k] for k in range(2)) if loaded else (0, 0)
            writer.writerow(dict(zip(fields, [t, action, count, *step, *offset,
                detail["total_us"], solver_time, detail["paper_eq3_same_segments_motion_us"], paper_time])))


def audit_circuit(nodes):
    checks = [defaultdict(set), defaultdict(set)]
    for node in nodes:
        if node.op.name == "cz":
            a, b = op_node_signature(node)[2]
            checks[int(a >= 216)][a].add(b)
    anticommuting = sum(len(x & z) % 2 for x in checks[0].values() for z in checks[1].values())
    return {"measurement_count": sum(n.op.name == "measure" for n in nodes),
            "odd_overlap_X_Z_check_pairs": anticommuting,
            "warning": "Treating the labeled X/Z neighborhoods as CSS checks gives odd overlaps; this QASM is not the paper's stabilizer measurement round. The supplied operations are preserved."}


def validate_raw_operations(qasm_path, schedule_path):
    """Also compare literal gate traces, independently of the DAG converter."""
    pattern = r"(?:h|z|cz) q\[\d+\](?:,q\[\d+\])?;"
    expected = re.findall(pattern, qasm_path.read_text())
    actual = re.findall(pattern, schedule_path.read_text())
    assert Counter(expected) == Counter(actual)
    for q in range(288):
        operand = f"q[{q}]"
        assert [g for g in expected if operand in g] == [g for g in actual if operand in g]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qasm", type=Path, default=ROOT / "inputs/qasm_files/bb144_bivariate_bicycle_cz.qasm")
    parser.add_argument("--coords", type=Path, default=ROOT.parent / "bb144_coords.json")
    parser.add_argument("--config", type=Path, default=ROOT / "inputs/algorithms/bb144_n_init.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/bb144_paper_optimized.schedule.txt")
    args = parser.parse_args()
    raw_config = json.loads(args.config.read_text())
    config = dict(raw_config)
    _coerce_unit_config(config)
    config["_motion_parameters"] = _motion_parameters(config)
    coords = json.loads(args.coords.read_text())
    qubits = initial_layout_fill(generate_grid(config["dimensions"], config["rydberg_radius"]), 288, coords)
    homes = {q.id: q.grid_position() for q in qubits}
    nodes = list(load_qasm_to_dag(args.qasm).topological_op_nodes())
    nodes = [n for n in nodes if n.op.name != "barrier"]
    assert all(n.op.name in {"h", "z", "cz"} for n in nodes)
    phase_nodes = [[n for n in nodes if n.op.name == "cz" and any(lo <= q < lo+72 for q in op_node_signature(n)[2])]
                   for lo in (144, 216)]
    assert [len(p) for p in phase_nodes] == [432, 432]
    # Emit the non-CZ operations as soon as wire order allows them, while
    # preserving the QASM's X-then-Z phase boundary.
    remaining = list(nodes)
    events, phases = [], []

    def singles_ready():
        blocked = set()
        ready = []
        for n in remaining:
            ids = set(op_node_signature(n)[2])
            if n.op.name != "cz" and not blocked & ids:
                ready.append(n)
            else:
                blocked.update(ids)
        ready_ids = {id(n) for n in ready}
        remaining[:] = [n for n in remaining if id(n) not in ready_ids]
        lines, _ = single_qubit_layer_time(ready, config["average_single_gate_time"], config["t_switch"])
        events.extend(("gate", line) for line in lines)

    singles_ready()
    for lo, phase in zip((144, 216), phase_nodes):
        emitted, info = emit_phase(phase, range(lo, lo+72), homes, config)
        events.extend(emitted)
        phases.append(info)
        phase_ids = {id(n) for n in phase}
        remaining[:] = [n for n in remaining if id(n) not in phase_ids]
        singles_ready()
    assert not remaining
    validate_schedule(events, homes, config, nodes)
    separate_time = schedule_duration(events, config)
    combined_events, combined_info = emit_combined(nodes, homes, config)
    validate_schedule(combined_events, homes, config, nodes)
    combined_time = schedule_duration(combined_events, config)
    if combined_time < separate_time:
        events, phases = combined_events, [combined_info]
    duration = schedule_duration(events, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_timed_schedule(args.output, solver="paper_ordered (naive_n_dag rules)",
                         qasm_filename=args.qasm.name, final_time=str(duration.to("microsecond")),
                         lattice_spacing=str(config["rydberg_radius"].to("micrometer")),
                         fill_seed=0, events=events, initial_qubits=qubits)
    read_homes, read_events = parse_serialized(args.output)
    assert read_homes == homes
    validate_schedule(read_events, read_homes, config, nodes)
    validate_raw_operations(args.qasm, args.output)
    assert abs((schedule_duration(read_events, config)-duration).to("second").magnitude) < 1e-12
    header_time = float(re.search(r"final_time: ([\d.e+-]+) microsecond", args.output.read_text())[1])
    assert abs(header_time - duration.to("microsecond").magnitude) < 1e-8
    write_timeline(args.output.with_suffix(".timeline.csv"), read_events, config, homes)
    report = {"qasm": str(args.qasm), "qasm_sha256": hashlib.sha256(args.qasm.read_bytes()).hexdigest(),
              "coords": str(args.coords), "config": raw_config,
              "operation_counts": dict(Counter(n.op.name for n in nodes)),
              "validation": "PASS: internal and serialized replay, full per-qubit operation order, original homes",
              "timing": timing_breakdown(events, config), "phases": phases,
              "candidate_us": {"separate_check_loads": separate_time.to("microsecond").magnitude,
                               "combined_check_load": combined_time.to("microsecond").magnitude},
              "circuit_audit": audit_circuit(nodes),
              "paper_comparison": "Eq. (3), PDF p. 6: sqrt(6*distance_um/0.02) per axis. The paper's 2.97 ms is a different circuit/layout/routing and motion model. Retiming these same segments is a comparison, not a reproduction of that result.",
              "optimality_scope": "Minimum motion within the enumerated separate/combined rigid check-load tours, one visit per displacement group and all input wire-order constraints; not global scheduling optimality. Gate pulse overhead does not enter the tour optimization."}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"schedule": str(args.output), "validation": report["validation"], "timing": report["timing"]}, indent=2))


if __name__ == "__main__":
    main()
