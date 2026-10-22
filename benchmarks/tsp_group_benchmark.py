#!/usr/bin/env python3
"""Benchmark displacement-tour ordering methods on synthetic AOD-load groups.

Compares, for one group (one AOD load) at a time:

  alignment        the per-layer alignment heuristic (no TSP)
  fast_tsp         relaxed side-priced TSP solved by fast_tsp
  concorde         relaxed side-priced TSP solved by Concorde
  gadget           exact Noon-Bean + Jonker-Volgenant gadget solved by Concorde

"Relaxed" prices every edge with its cheapest pair of interaction sides, so a
tour may arrive at a city on one side and leave from another. The gadget
makes each city a cluster of its four side sites and converts that
generalized TSP into a symmetric TSP, so its optimum is an optimal order and
side choice.

Every method only produces a visit ORDER. Each order is then timed the same
way the scheduler times it: ``_optimize_visit_sides`` picks the best
interaction site for every gate and returns the exact parity-routed motion
time. So "quality" below is always real motion time, comparable across
methods.

Two things are measured:

  1. Execution time by group size: graph build time (shared by the solvers
     that use the same graph) plus solve time, for every size in --sizes.
  2. Quality as a function of time: fast_tsp, concorde and gadget are rerun
     at every limit in --time-limits, so you can see how the motion time
     improves as each solver gets more time. Alignment has no time limit.

Groups are synthetic, not taken from a circuit: ``random.Random(seed)``
draws ``size`` even displacements in [-2*span, 2*span]^2 (duplicates are
dropped, so the city count can come out slightly below ``size``), optionally
spread over several layers. Within one (size, seed), every method solves the
same instance.

Usage (from the repository root, with the package installed):

    python benchmarks/tsp_group_benchmark.py \\
        --sizes 50 200 500 1000 --time-limits 0.1 1 10 60 \\
        --csv benchmarks/output/results/tsp_group_benchmark.csv

Quick smoke test (seconds):

    python benchmarks/tsp_group_benchmark.py \\
        --sizes 20 --time-limits 0.1

Rough costs, measured 2026-10-17 on one core (single layer, Cubic_Spline):

  - Relaxed graph build is O(n^2) Python: ~4 s at 200 cities, ~22 s at 500,
    ~85 s at 1000.
  - Gadget build is ~8x the nodes: ~7 s at 200, ~41 s at 500, ~146 s at 1000,
    then Concorde usually runs to its full limit (the limit is soft and can
    overrun). With limits 0.1 1 10 60 the gadget alone costs over a minute
    per size. Use --gadget-max-cities to skip it on large groups.
  - Each Concorde solve runs in a forked child process (gadget) or in
    process (relaxed); memory at 1000 cities is a few GB for the gadget.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import warnings
from pathlib import Path

# Make ``naive_n_dag`` importable without setting PYTHONPATH.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from naive_n_dag import dynamics as d  # noqa: E402
from naive_n_dag import tsp_gadget as tg  # noqa: E402
from naive_n_dag.main import _coerce_unit_config  # noqa: E402

# Physics used when --config is not given: the fast-motion settings of the
# qLDPC configs (8 um lattice, 10 m/s, 20000 m/s^2, Cubic_Spline moves).
DEFAULT_PHYSICS = dict(
    transfer_SLM_AOD="15 microseconds",
    max_acceleration="20000 m/s^2",
    max_velocity="10 m/s",
    lattice_spacing="8 micrometers",
    average_single_gate_time="0.1 microseconds",
    average_two_gate_time="0.27 microseconds",
    measurement_time="0 microseconds",
    t_switch="1.5 microseconds",
    movement_shape="Cubic_Spline",
    alignment_conc=-1,
    max_dimension=[24, 12],
    T_reuse="Inf",
)

# Keys copied from a --config file; everything else in it (layouts, QASM
# paths, fill strategy, ...) is irrelevant to a single-group benchmark.
PHYSICS_KEYS = (
    "transfer_SLM_AOD", "max_acceleration", "max_velocity", "lattice_spacing",
    "average_single_gate_time", "average_two_gate_time", "measurement_time",
    "t_switch", "movement_shape",
)

METHODS = ("alignment", "fast_tsp", "concorde", "gadget")

CSV_FIELDS = (
    "size", "cities", "layers", "seed", "start_offset", "method", "time_limit_s",
    "build_s", "solve_s", "total_s", "status", "motion_us", "vs_alignment_pct",
    "vs_best_pct",
)


def load_physics(config_path, movement_shape):
    """Unit-coerced physics config, from a repo JSON config or the defaults."""
    config = dict(DEFAULT_PHYSICS)
    if config_path:
        raw = json.loads(Path(config_path).read_text())
        config.update({key: raw[key] for key in PHYSICS_KEYS if key in raw})
    if movement_shape:
        config["movement_shape"] = movement_shape
    _coerce_unit_config(config)
    # The scheduler caches these once per run; do the same here.
    config["_motion_parameters"] = d._motion_parameters(config)
    return config


def make_instance(size, seed, span, layers):
    """Distinct (layer, displacement) cities for one synthetic group.

    Displacements are even-even (home-relative gate partners). Layers are
    drawn uniformly, then renumbered 0..L-1 so no layer is empty.
    """
    rng = random.Random(seed)
    cities = list(dict.fromkeys(
        (rng.randrange(layers), (2 * rng.randint(-span, span), 2 * rng.randint(-span, span)))
        for _ in range(size)
    ))
    rank = {layer: i for i, layer in enumerate(sorted({layer for layer, _ in cities}))}
    return [(rank[layer], vector) for layer, vector in cities]


def run_instance(size, seed, args, config, emit):
    """Run every requested method on one instance; ``emit`` receives result rows."""
    cities = make_instance(size, seed, args.span, args.layers)
    if len(cities) < 2:
        print(f"size={size} seed={seed}: fewer than 2 distinct cities, skipped")
        return
    start = tuple(args.start_offset)
    vectors = [vector for _, vector in cities]
    labels = [layer for layer, _ in cities]
    base = dict(size=size, cities=len(cities), layers=len(set(labels)), seed=seed,
                start_offset=f"{start[0]},{start[1]}")

    def seconds(order):
        """Exact motion time of a city order: the side DP picks the sites."""
        return d._city_order_seconds([cities[c][1] for c in order], config, start)

    def to_city_order(vector_order):
        index = {city: i for i, city in enumerate(cities)}
        return list(dict.fromkeys(index[(labels[i], vectors[i])] for i in vector_order))

    rows = []

    def record(method, limit, build_s, solve_s, status, motion=None):
        rows.append(dict(base, method=method, time_limit_s=limit, build_s=round(build_s, 3),
                         solve_s=round(solve_s, 3), total_s=round(build_s + solve_s, 3),
                         status=status, motion=motion))

    # --- alignment: no graph, no time limit. Its time is also the baseline.
    t = time.perf_counter()
    alignment = to_city_order(d._alignment_order(vectors, labels, start))
    alignment_s = time.perf_counter() - t
    alignment_motion = seconds(alignment)
    record("alignment", "", 0.0, alignment_s, "ok", alignment_motion)
    best_motion = alignment_motion  # best feasible tour so far; bounds the gadget's K

    concorde, fast_tsp = d._tsp_backends()

    # --- relaxed side-priced graph, built once and shared by fast_tsp and Concorde.
    relaxed_methods = [m for m in ("fast_tsp", "concorde") if m in args.methods]
    if relaxed_methods:
        t = time.perf_counter()
        graph = d._layered_tour_graph(cities, config, start)
        graph_s = time.perf_counter() - t
        for method in relaxed_methods:
            for limit in args.time_limits:
                if method == "fast_tsp":
                    if fast_tsp is None:
                        record(method, limit, graph_s, 0.0, "unavailable")
                        continue
                    solve = lambda w: d._fast_tsp_tour(w, fast_tsp, limit, graph.tour_bound)
                else:
                    if concorde is None:
                        record(method, limit, graph_s, 0.0, "unavailable")
                        continue
                    solve = lambda w: d._concorde_tour(w, *concorde, limit, graph.tour_bound)
                t = time.perf_counter()
                try:
                    # The validator rejects tours that break layer order or use absent edges.
                    order = d._validated_layer_order(solve(graph.weights), graph)
                    solve_s = time.perf_counter() - t
                    motion = seconds(order)
                    best_motion = min(best_motion, motion)
                    record(method, limit, graph_s, solve_s, "ok", motion)
                except Exception as exc:
                    record(method, limit, graph_s, time.perf_counter() - t, f"failed: {exc}")

    # --- gadget: exact formulation, built once, solved by Concorde per limit.
    if "gadget" in args.methods:
        if concorde is None:
            record("gadget", "", 0.0, 0.0, "unavailable")
        elif args.gadget_max_cities is not None and len(cities) > args.gadget_max_cities:
            record("gadget", "", 0.0, 0.0, f"skipped (> {args.gadget_max_cities} cities)")
        else:
            t = time.perf_counter()
            try:
                # K must exceed the optimal route total; the best tour found so
                # far is a valid upper bound (the scheduler uses the relaxed one).
                gadget = tg.build_gadget(
                    cities, start,
                    lambda a, b: d._route(a, b, config)[0],
                    lambda step: d._step_seconds(step, config),
                    best_motion,
                )
            except Exception as exc:
                gadget = None
                record("gadget", "", time.perf_counter() - t, 0.0, f"build failed: {exc}")
            gadget_s = time.perf_counter() - t
            for limit in args.time_limits if gadget is not None else ():
                t = time.perf_counter()
                try:
                    # Forked child: a Concorde abort only loses this result.
                    tour = d._in_child_process(d._concorde_integer_tour, gadget.weights, *concorde, limit)
                    solve_s = time.perf_counter() - t
                    # Rejects split clusters, reversed passes, backward layers, absent edges.
                    order, _sites = tg.decode_gadget_tour(tour, gadget)
                    record("gadget", limit, gadget_s, solve_s, "ok", seconds(order))
                except Exception as exc:
                    record("gadget", limit, gadget_s, time.perf_counter() - t, f"rejected: {exc}")

    # Percentages need the best motion over all methods, so emit at the end.
    best = min(row["motion"] for row in rows if row["motion"] is not None)
    for row in rows:
        motion = row.pop("motion")
        row["motion_us"] = "" if motion is None else round(motion * 1e6, 3)
        row["vs_alignment_pct"] = "" if motion is None else round(100 * (motion / alignment_motion - 1), 3)
        row["vs_best_pct"] = "" if motion is None else round(100 * (motion / best - 1), 3)
        emit(row)


def print_summary(rows, time_limits):
    """Per instance: one line per method, one column per time limit.

    Each cell is "motion change vs alignment / wall seconds (build + solve)".
    Read across a row for quality versus time; read down a column to compare
    methods at the same budget.
    """
    instances = {}
    for row in rows:
        instances.setdefault((row["size"], row["seed"]), []).append(row)
    for (size, seed), group in instances.items():
        first = group[0]
        print(f"\n== size {size} (cities {first['cities']}, layers {first['layers']}, "
              f"seed {seed}, start {first['start_offset']}) ==")
        header = f"{'method':<10}" + "".join(f"{f'{lim}s limit':>26}" for lim in time_limits)
        print(header)
        for method in METHODS:
            cells = {row["time_limit_s"]: row for row in group if row["method"] == method}
            if not cells:
                continue
            if method == "alignment" or "" in cells:
                row = next(iter(cells.values()))
                cell = (f"{row['vs_alignment_pct']:+.2f}% / {row['total_s']:.2f}s"
                        if row["status"] == "ok" else row["status"])
                print(f"{method:<10}{cell:>26}")
                continue
            line = f"{method:<10}"
            for limit in time_limits:
                row = cells.get(limit)
                if row is None:
                    cell = "-"
                elif row["status"] == "ok":
                    cell = f"{row['vs_alignment_pct']:+.2f}% / {row['total_s']:.1f}s"
                else:
                    cell = row["status"].split(":")[0]
                line += f"{cell:>26}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 200, 500],
                        help="requested cities per group (default: 50 200 500)")
    parser.add_argument("--time-limits", type=float, nargs="+", default=[0.1, 1.0, 10.0],
                        help="seconds per solve for fast_tsp/concorde/gadget (default: 0.1 1 10)")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS),
                        help="methods to run (alignment always runs: it is the baseline)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="instance seeds per size (default: the size itself)")
    parser.add_argument("--layers", type=int, default=1,
                        help="layers per group; cities are spread over them (default: 1)")
    parser.add_argument("--span", type=int, default=60,
                        help="displacements are even values in [-2*span, 2*span] (default: 60)")
    parser.add_argument("--start-offset", type=int, nargs=2, default=[0, 0], metavar=("ROW", "COL"),
                        help="load start: 0 0 = from home; an AOD offset (odd sum) = retained load")
    parser.add_argument("--gadget-max-cities", type=int, default=None,
                        help="skip the gadget above this many cities (default: never skip)")
    parser.add_argument("--config", default=None,
                        help="take physics (velocity, acceleration, lattice, shape, ...) from a repo JSON config")
    parser.add_argument("--movement-shape", choices=["Cubic_Spline", "trapezoidal"], default=None,
                        help="override the movement profile")
    parser.add_argument("--csv", default=None,
                        help="also write every result row to this CSV file (written as rows finish)")
    args = parser.parse_args()
    args.methods = ["alignment", *[m for m in args.methods if m != "alignment"]]

    # Solver fallbacks warn; the benchmark records failures in the status column instead.
    warnings.simplefilter("ignore")
    config = load_physics(args.config, args.movement_shape)

    rows = []
    writer = handle = None
    if args.csv:
        handle = open(args.csv, "w", newline="")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

    def emit(row):
        rows.append(row)
        if writer:
            writer.writerow(row)
            handle.flush()  # keep partial results if a long run is interrupted

    try:
        for size in args.sizes:
            for seed in args.seeds or [size]:
                t = time.perf_counter()
                run_instance(size, seed, args, config, emit)
                print(f"size={size} seed={seed} done in {time.perf_counter() - t:.1f}s", flush=True)
    finally:
        if handle:
            handle.close()
    print_summary(rows, args.time_limits)


if __name__ == "__main__":
    main()
