#!/usr/bin/env python3
"""Which placement cost is better: F ("fastsa") or F_time ("time")?

F is no longer in src/ (F_time replaced it on 2026-10-20); this script takes it
from Legacy/legacy_f.py and hands each cost to the placers through their
``cost=`` argument, so the comparison stays reproducible.

The scheduler is the judge. It is run only here, offline, to time layouts;
neither placer calls it. For every (circuit, physics) job:

1. OUTCOME. Optimise with each cost: the deterministic placer (reproducible;
   it only compares costs, so it is the clean test) and FastSA (seeds
   0..k-1, main's default settings; FastSA normalises each cost by its mean
   random-layout value, so both costs are annealed on the same scale). Then
   schedule each result.
2. FIDELITY. Score a sample of layouts under both costs and schedule each:
     random     uniformly random layouts
     seed       structured seeds (graph orders x paths, row-major)
     optimum    the outcome layouts above
     neighbour  small moves from the best layouts ("anchors"): random swaps,
                swaps with a nearby qubit, relocations to a free site
   Three metrics per cost:
     spearman     rank correlation of cost and scheduled time over the sample
     local_agree  over (anchor, neighbour) pairs whose scheduled times differ
                  by more than 0.1%, the share where the cost moves in the
                  same direction. A search only ever compares neighbours, so
                  this decides whether a cost guides it correctly.
     regret       scheduled time of the sample's best layout BY COST over the
                  sample's best layout by scheduled time, minus 1
3. EASE. Milliseconds per evaluation and optimisation wall time.

Circuits are general, not structured codes: QFT-18/29, BV-50 and VQE-30 from
the repo, plus generated QAOA, random two-qubit, quantum-volume,
Heisenberg-Trotter and ripple-carry-adder circuits (written once to
inputs/qasm_files/general/). Two physics settings change which time term dominates:
"slow" (QFT-29 config: 0.55 m/s, motion-dominated) and "fast" (bb144 config:
10 m/s). The scheduler uses tour_ordering "tsp" throughout.

Usage (from the repo root):

    python Legacy/cost_fidelity.py \\
        --workers 10 --csv benchmarks/output/results/cost_fidelity.csv

Quick check: --circuits qft_18 --physics slow --sample-scale 0.2 --fastsa-seeds 1
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
CIRCUIT_DIR = REPO / "inputs" / "qasm_files" / "general"

PHYSICS = {
    # Slow motion (QFT-29 config's 0.55 m/s, 2750 m/s^2), so travel dominates.
    "slow": dict(
        transfer_SLM_AOD="15 microseconds", max_acceleration="2750 m/s^2", max_velocity="0.55 m/s",
        lattice_spacing="15 micrometers", average_single_gate_time="1 microseconds",
        average_two_gate_time="0.27 microseconds", t_switch="1.5 microseconds",
        movement_shape="trapezoidal", T_reuse=3,
    ),
    # Fast motion (bb144 config's 10 m/s, 20000 m/s^2, Cubic_Spline), so transfers and pulses weigh more.
    "fast": dict(
        transfer_SLM_AOD="15 microseconds", max_acceleration="20000 m/s^2", max_velocity="10 m/s",
        lattice_spacing="8 micrometers", average_single_gate_time="0.1 microseconds",
        average_two_gate_time="0.27 microseconds", measurement_time="0 microseconds",
        t_switch="1.5 microseconds", movement_shape="Cubic_Spline", T_reuse="Inf",
    ),
}

REPO_CIRCUITS = {
    "qft_18": REPO / "inputs/qasm_files/qft/qft_n18_transpiled.qasm",
    "qft_29": REPO / "inputs/qasm_files/qft/qft_n29_transpiled.qasm",
    "bv_50": REPO / "inputs/qasm_files/bv/bv_n50.qasm",
    "vqe_30": REPO / "inputs/qasm_files/vqe/vqe_n30.qasm",
}
GENERATED = ("qaoa_24", "random_24", "qv_20", "heisenberg_24", "adder_18")
CIRCUITS = (*REPO_CIRCUITS, *GENERATED)
COSTS = ("fastsa", "time")
COST_COLUMN = {"fastsa": "F", "time": "F_time"}
# FastSA settings: naive_n_dag.main's current Fastsa_fill defaults. The original
# 2026-10-20 comparison used P = 0.99 and threshold 1e-3, the defaults then.
FASTSA_SETTINGS = dict(stage1_iterations=200, stage2_iterations=100, initial_accept_prob=0.999,
                       c=100.0, stage3_temperature_threshold=0.0, stage3_max_iterations=1000,
                       stage3_section_size=4)


def make_costs(ctx):
    """Both costs for one job: the legacy F and F_time."""
    from legacy_f import FastSACost
    from naive_n_dag.placement_cost import make_cost
    return {"fastsa": FastSACost(ctx["dag"], ctx["n"]),
            "time": make_cost(ctx["dag"], ctx["n"], ctx["config"], ctx["reference"])}


# ---------------------------------------------------------------------------
# Circuits
# ---------------------------------------------------------------------------

def _regular_graph(n, degree, seed):
    """Sorted edges of a seeded random ``degree``-regular graph on ``n`` nodes."""
    import networkx as nx
    return sorted(tuple(sorted(e)) for e in nx.random_regular_graph(degree, n, seed=seed).edges())


def _generate(name):
    """One generated benchmark circuit, transpiled to {cx, rz, sx, x}."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import CDKMRippleCarryAdder, QuantumVolume

    rng = random.Random(7)
    if name == "qaoa_24":                         # MaxCut QAOA, p = 2, random 3-regular graph
        n, edges = 24, _regular_graph(24, 3, seed=7)
        qc = QuantumCircuit(n)
        qc.h(range(n))
        for _ in range(2):
            for u, v in edges:
                qc.rzz(rng.uniform(0, math.pi), u, v)
            for q in range(n):
                qc.rx(rng.uniform(0, math.pi), q)
    elif name == "random_24":                     # 150 random CX pairs with 1Q rotations
        n = 24
        qc = QuantumCircuit(n)
        for _ in range(150):
            a, b = rng.sample(range(n), 2)
            qc.rz(rng.uniform(0, math.pi), a)
            qc.cx(a, b)
    elif name == "qv_20":                         # quantum volume, 20 qubits, depth 6
        qc = QuantumVolume(20, depth=6, seed=7).decompose()
    elif name == "heisenberg_24":                 # 3 Trotter steps of XX+YY+ZZ on a 3-regular graph
        n, edges = 24, _regular_graph(24, 3, seed=11)
        qc = QuantumCircuit(n)
        for _ in range(3):
            for u, v in edges:
                qc.rxx(0.3, u, v)
                qc.ryy(0.3, u, v)
                qc.rzz(0.3, u, v)
    elif name == "adder_18":                      # 8-bit CDKM ripple-carry adder (18 qubits)
        qc = CDKMRippleCarryAdder(8, kind="full").decompose().decompose()
    else:
        raise ValueError(name)
    qc = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    # One register: the repo's loader reads qubit indices per register, so a
    # multi-register circuit (the adder's a, b, cin, cout) would alias qubits.
    flat = QuantumCircuit(qc.num_qubits)
    flat.compose(qc, qubits=list(range(qc.num_qubits)), inplace=True)
    return flat


def circuit_path(name):
    """QASM path for a benchmark circuit, writing generated ones on first use."""
    if name in REPO_CIRCUITS:
        return REPO_CIRCUITS[name]
    path = CIRCUIT_DIR / f"{name}.qasm"
    if not path.exists():
        from qiskit import qasm2
        CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(qasm2.dumps(_generate(name)))
    return path


# ---------------------------------------------------------------------------
# One job: (circuit, physics)
# ---------------------------------------------------------------------------

def _neighbours(layout, rng, R, C, count):
    """Small moves from a layout: random swaps, near swaps, relocations."""
    n = len(layout)
    free = sorted({(r, c) for r in range(R) for c in range(C)} - set(layout))
    out = []
    for k in range(count):
        trial = list(layout)
        move = ("swap", "near", "relocate")[k % 3]
        if move == "relocate" and not free:
            move = "swap"                                       # a full array has no free site
        if move == "swap":                                      # random swap
            i, j = rng.sample(range(n), 2)
            trial[i], trial[j] = trial[j], trial[i]
        elif move == "near":                                    # swap with a nearby qubit
            i = rng.randrange(n)
            near = sorted(range(n), key=lambda j: ((layout[j][0] - layout[i][0]) ** 2
                                                   + (layout[j][1] - layout[i][1]) ** 2, j))[1:4]
            j = rng.choice(near)
            trial[i], trial[j] = trial[j], trial[i]
        else:                                                   # relocate to a free site
            i = rng.randrange(n)
            trial[i] = rng.choice(free)
        out.append((move, trial))
    return out


def job_context(name, physics):
    """Load a circuit under one physics setting; return what placing and timing need."""
    warnings.simplefilter("ignore")
    from naive_n_dag.dag_helper import load_qasm_to_two_qubit_dag_with_single_qubit_context as load
    from naive_n_dag.dynamics import schedule_circuit
    from naive_n_dag.grid import generate_grid, place_qubit
    from naive_n_dag.main import _coerce_unit_config

    dag, singles, reference = load(circuit_path(name), return_reference_nodes=True)
    n = len(dag.qubits)
    rows = math.isqrt(n) + (math.isqrt(n) ** 2 < n)
    dims = [rows, -(-n // rows)]
    config = dict(PHYSICS[physics], dimensions=dims, max_dimension=list(dims), num_NA=n,
                  alignment_conc=-1, tour_ordering="tsp")
    _coerce_unit_config(config)
    if config["T_reuse"] == "Inf":
        config["T_reuse"] = float("inf")
    layers = [[node for node in layer["graph"].op_nodes() if node.op.name != "barrier"]
              for layer in dag.layers()]
    cache = {}

    def texec(layout):
        """Schedule a layout of (row, col) sites with the real scheduler; microseconds."""
        key = tuple(layout)
        if key not in cache:
            grid = generate_grid(dims, config["lattice_spacing"])
            qubits = [place_qubit(grid, 2 * r, 2 * c, q) for q, (r, c) in enumerate(layout)]
            schedule = schedule_circuit(layers, singles, qubits, config, reference)
            cache[key] = schedule.duration.to("microsecond").magnitude
        return cache[key]

    return dict(dag=dag, singles=singles, reference=reference, n=n, dims=dims,
                config=config, layers=layers, texec=texec, cache=cache)


def run_job(job):
    """Run OUTCOME, FIDELITY and EASE for one (circuit, physics) job.

    Returns ``(summary, records)``: one row of metrics and every scored layout.
    """
    name, physics, budget, fastsa_seeds, scale = job
    warnings.simplefilter("ignore")
    from naive_n_dag import deterministic_placement as dp
    from naive_n_dag.FastSA_Solver import run_fastsa
    from naive_n_dag.grid import generate_grid

    started = time.time()
    ctx = job_context(name, physics)
    dag, reference, n, dims = ctx["dag"], ctx["reference"], ctx["n"], ctx["dims"]
    config, layers, texec, texec_cache = ctx["config"], ctx["layers"], ctx["texec"], ctx["cache"]
    R, C = dims
    costs = make_costs(ctx)

    # Evaluation speed of each cost (EASE), on random layouts.
    rng = random.Random(f"{name}/{physics}")
    sites = [(r, c) for r in range(R) for c in range(C)]
    probe = [rng.sample(sites, n) for _ in range(20)]
    eval_ms = {}
    for c, cost in costs.items():
        t = time.perf_counter()
        for layout in probe:
            cost.evaluate(layout)
        eval_ms[c] = 1e3 * (time.perf_counter() - t) / len(probe)

    records = []

    def record(kind, label, layout, **extra):
        layout = [tuple(site) for site in layout]
        row = dict(circuit=name, physics=physics, n=n, gates=sum(len(l) for l in layers),
                   kind=kind, label=label, F=costs["fastsa"].evaluate(layout),
                   F_time=costs["time"].evaluate(layout), texec=texec(layout), layout=layout)
        row.update(extra)
        records.append(row)
        return row

    # 1. OUTCOME
    for cost in COSTS:
        t = time.time()
        result = dp.run_deterministic_placement(dag, n, dims, reference_nodes=reference,
                                                options={"budget": budget}, cost=costs[cost])
        record("optimum", f"deterministic/{cost}", result.layout, optimiser="deterministic",
               cost=cost, wall_s=time.time() - t, evaluations=result.evaluations)
        for seed in range(fastsa_seeds):
            t = time.time()
            r = run_fastsa(generate_grid(dims, 1.0), n, dag, config, seed=seed, reference_nodes=reference,
                           cost=costs[cost], **FASTSA_SETTINGS)
            layout = [(r.best_positions[q][1] // 2, r.best_positions[q][0] // 2) for q in range(n)]
            record("optimum", f"fastsa/{cost}/seed{seed}", layout, optimiser="fastsa",
                   cost=cost, wall_s=time.time() - t)

    # 2. FIDELITY sample
    for i in range(max(2, round(20 * scale))):
        record("random", f"random{i}", rng.sample(sites, n))
    seeds = dp._general_seeds(costs["fastsa"].layer_pairs, n, R, C)
    step = max(1, len(seeds) // max(2, round(20 * scale)))
    seed_rows = [record("seed", label, layout) for label, layout in seeds[::step]]
    seed_rows.append(record("seed", "row-major", [divmod(i, C) for i in range(n)]))
    anchors = [r for r in records if r["kind"] == "optimum"]
    for column in COST_COLUMN.values():
        anchors += sorted(seed_rows, key=lambda r: r[column])[:2]
    unique = list({tuple(a["layout"]): a for a in anchors}.values())
    pairs = []
    for anchor in unique:
        for move, layout in _neighbours(anchor["layout"], rng, R, C, max(3, round(6 * scale))):
            pairs.append((anchor, record("neighbour", f"{anchor['label']}+{move}", layout, move=move)))

    # Metrics
    summary = dict(circuit=name, physics=physics, n=n, gates=records[0]["gates"],
                   layouts=len(texec_cache), routes_s=round(time.time() - started, 1))
    texecs = [r["texec"] for r in records]
    best_texec = min(texecs)
    for cost, column in COST_COLUMN.items():
        values = [r[column] for r in records]
        from scipy.stats import spearmanr
        summary[f"spearman_{cost}"] = float(spearmanr(values, texecs).statistic)
        agree = total = 0
        for anchor, neighbour in pairs:
            d_t = neighbour["texec"] - anchor["texec"]
            if abs(d_t) <= 1e-3 * anchor["texec"]:
                continue
            d_c = neighbour[column] - anchor[column]
            total += 1
            agree += (d_c > 0) == (d_t > 0) and d_c != 0
        summary[f"local_agree_{cost}"] = agree / total if total else float("nan")
        summary["local_pairs"] = total
        pick = min(records, key=lambda r: (r[column], r["texec"]))
        summary[f"regret_{cost}"] = pick["texec"] / best_texec - 1
        summary[f"eval_ms_{cost}"] = eval_ms[cost]
        det = next(r for r in records if r["label"] == f"deterministic/{cost}")
        summary[f"det_texec_{cost}"] = det["texec"]
        summary[f"det_wall_{cost}"] = det["wall_s"]
        runs = [r for r in records if r.get("optimiser") == "fastsa" and r.get("cost") == cost]
        summary[f"fastsa_median_{cost}"] = (
            statistics.median(r["texec"] for r in runs) if runs else float("nan"))
        summary[f"fastsa_wall_{cost}"] = (
            statistics.mean(r["wall_s"] for r in runs) if runs else float("nan"))
    summary["best_sampled_texec"] = best_texec
    summary["total_s"] = round(time.time() - started, 1)
    for r in records:
        r.pop("layout")
    return summary, records


# ---------------------------------------------------------------------------

def _fmt(x, digits=3):
    return "nan" if x != x else f"{x:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--circuits", nargs="+", default=list(CIRCUITS), choices=CIRCUITS)
    parser.add_argument("--physics", nargs="+", default=list(PHYSICS), choices=list(PHYSICS))
    parser.add_argument("--budget", type=int, default=20_000,
                        help="deterministic placer's evaluation budget (same for both costs)")
    parser.add_argument("--fastsa-seeds", type=int, default=3)
    parser.add_argument("--sample-scale", type=float, default=1.0,
                        help="scale the fidelity sample (1.0: ~20 random, ~20 seeds, 6 neighbours per anchor)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--csv", default=None, help="write every scored layout to this CSV")
    args = parser.parse_args()

    for name in args.circuits:                    # generate once, before forking
        circuit_path(name)
    jobs = [(c, p, args.budget, args.fastsa_seeds, args.sample_scale)
            for c in args.circuits for p in args.physics]
    summaries, rows = [], []
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for summary, records in pool.imap_unordered(run_job, jobs):
            summaries.append(summary)
            rows.extend(records)
            print(f"done {summary['circuit']}/{summary['physics']}: {summary['layouts']} layouts "
                  f"in {summary['total_s']}s", flush=True)

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    summaries.sort(key=lambda s: (s["physics"], s["circuit"]))
    print("\nFIDELITY (higher spearman / local_agree is better; regret: lower is better)")
    print(f"{'job':22s} {'pairs':>5s} | {'rho F':>6s} {'rho Ft':>6s} | {'loc F':>6s} {'loc Ft':>6s} | "
          f"{'reg F':>6s} {'reg Ft':>6s} | {'ms F':>6s} {'ms Ft':>6s}")
    for s in summaries:
        print(f"{s['circuit'] + '/' + s['physics']:22s} {s['local_pairs']:5d} | "
              f"{_fmt(s['spearman_fastsa'], 2):>6s} {_fmt(s['spearman_time'], 2):>6s} | "
              f"{_fmt(s['local_agree_fastsa'], 2):>6s} {_fmt(s['local_agree_time'], 2):>6s} | "
              f"{_fmt(100 * s['regret_fastsa'], 1):>5s}% {_fmt(100 * s['regret_time'], 1):>5s}% | "
              f"{_fmt(s['eval_ms_fastsa'], 2):>6s} {_fmt(s['eval_ms_time'], 2):>6s}")
    print("\nOUTCOME (scheduled time in us; lower is better)")
    print(f"{'job':22s} | {'det F':>10s} {'det Ft':>10s} | {'SA F':>10s} {'SA Ft':>10s} | "
          f"{'det s F':>7s} {'det s Ft':>8s} {'SA s F':>7s} {'SA s Ft':>7s}")
    for s in summaries:
        print(f"{s['circuit'] + '/' + s['physics']:22s} | {s['det_texec_fastsa']:10.1f} {s['det_texec_time']:10.1f} | "
              f"{s['fastsa_median_fastsa']:10.1f} {s['fastsa_median_time']:10.1f} | "
              f"{s['det_wall_fastsa']:7.1f} {s['det_wall_time']:8.1f} "
              f"{s['fastsa_wall_fastsa']:7.1f} {s['fastsa_wall_time']:7.1f}")

    def agg(key):
        values = [s[key] for s in summaries if s[key] == s[key]]
        return statistics.median(values) if values else float("nan")

    wins = lambda a, b: sum(s[a] < s[b] - 1e-9 for s in summaries)
    print("\nAGGREGATE over", len(summaries), "jobs (medians)")
    for metric in ("spearman", "local_agree", "regret", "eval_ms"):
        print(f"  {metric:12s} F={_fmt(agg(metric + '_fastsa'))}  F_time={_fmt(agg(metric + '_time'))}")
    print(f"  deterministic: F_time faster schedule in {wins('det_texec_time', 'det_texec_fastsa')}, "
          f"F faster in {wins('det_texec_fastsa', 'det_texec_time')} jobs")
    print(f"  FastSA median: F_time faster schedule in {wins('fastsa_median_time', 'fastsa_median_fastsa')}, "
          f"F faster in {wins('fastsa_median_fastsa', 'fastsa_median_time')} jobs")
    for label, a, b in (("deterministic", "det_texec_time", "det_texec_fastsa"),
                        ("FastSA", "fastsa_median_time", "fastsa_median_fastsa")):
        ratio = [s[a] / s[b] for s in summaries]
        geo = math.exp(statistics.mean(math.log(x) for x in ratio))
        print(f"  {label} F_time/F scheduled-time ratio: geomean {_fmt(geo)}, median "
              f"{_fmt(statistics.median(ratio))}, min {_fmt(min(ratio))}, max {_fmt(max(ratio))}")


if __name__ == "__main__":
    main()
