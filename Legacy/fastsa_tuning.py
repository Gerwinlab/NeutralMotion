#!/usr/bin/env python3
"""Tune FastSA's schedule (P and the stage-3 stopping threshold) under cost normalisation.

FastSA now divides its cost by the mean random-layout cost (Chen & Chang's
normalisation), so P (``initial_accept_prob``) and ``stage3_temperature_threshold``
mean the same thing on every circuit and can be tuned once. This script:

1. runs FastSA for every (cost, P, threshold) in the grid, on every
   (circuit, physics) job of benchmarks/cost_fidelity.py, with seeds 0..k-1,
   and schedules each result with the real scheduler (offline only);
2. picks each cost's best setting on the TRAINING jobs only: lowest geometric
   mean of (median scheduled time over seeds) / (old shipped FastSA's median,
   read from the earlier cost_fidelity CSV, i.e. F unnormalised with
   P = 0.99 and threshold 1e-3);
3. reports those settings on the HELD-OUT jobs, so the chosen defaults are not
   fitted to the circuits they are judged on.

Usage (from the repo root):

    ~/Quantum_Class/Phys765_Test/bin/python Legacy/fastsa_tuning.py \\
        --reference benchmarks/output/results/cost_fidelity_2026-10-20.csv \\
        --csv benchmarks/output/results/fastsa_tuning_2026-10-20.csv --workers 10
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import math
import multiprocessing
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_fidelity as cf  # noqa: E402

TRAIN = ("qft_18", "qaoa_24", "random_24", "adder_18", "vqe_30")
HELD_OUT = ("qft_29", "bv_50", "qv_20", "heisenberg_24")
P_GRID = (0.99, 0.999, 0.9999, 0.99999)
THRESHOLD_GRID = (1e-3, 0.0)        # 0: stage 3 always runs stage3_max_iterations


def run_job(job):
    name, physics, seeds = job
    warnings.simplefilter("ignore")
    from naive_n_dag.FastSA_Solver import run_fastsa
    from naive_n_dag.grid import generate_grid

    ctx = cf.job_context(name, physics)
    costs = cf.make_costs(ctx)
    rows = []
    for cost, p, threshold in itertools.product(cf.COSTS, P_GRID, THRESHOLD_GRID):
        settings = dict(cf.FASTSA_SETTINGS, initial_accept_prob=p, stage3_temperature_threshold=threshold)
        for seed in range(seeds):
            t = time.time()
            r = run_fastsa(generate_grid(ctx["dims"], 1.0), ctx["n"], ctx["dag"], ctx["config"],
                           seed=seed, reference_nodes=ctx["reference"], cost=costs[cost], **settings)
            wall = time.time() - t
            layout = [(r.best_positions[q][1] // 2, r.best_positions[q][0] // 2) for q in range(ctx["n"])]
            rows.append(dict(circuit=name, physics=physics, cost=cost, P=p, threshold=threshold,
                             seed=seed, texec=ctx["texec"](layout), wall_s=wall,
                             stage3_iterations=r.stage3_iterations))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True,
                        help="cost_fidelity CSV holding the old shipped FastSA runs (label fastsa-abs/fastsa/...)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    reference = collections.defaultdict(list)
    for r in csv.DictReader(open(args.reference)):
        if r["kind"] == "optimum" and r["label"].startswith("fastsa-abs/fastsa/"):
            reference[(r["circuit"], r["physics"])].append(float(r["texec"]))
    reference = {k: statistics.median(v) for k, v in reference.items()}

    jobs = [(c, p, args.seeds) for c in (*TRAIN, *HELD_OUT) for p in cf.PHYSICS]
    rows = []
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for job_rows in pool.imap_unordered(run_job, jobs):
            rows.extend(job_rows)
            print(f"done {job_rows[0]['circuit']}/{job_rows[0]['physics']}", flush=True)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    # Median scheduled time over seeds, per (job, cost, P, threshold).
    median = collections.defaultdict(list)
    for r in rows:
        median[(r["circuit"], r["physics"], r["cost"], r["P"], r["threshold"])].append(r["texec"])
    median = {k: statistics.median(v) for k, v in median.items()}
    stage3 = collections.defaultdict(list)
    for r in rows:
        stage3[(r["cost"], r["P"], r["threshold"])].append(r["stage3_iterations"])

    def geomean_ratio(circuits, cost, p, threshold):
        ratios = [median[(c, ph, cost, p, threshold)] / reference[(c, ph)]
                  for c in circuits for ph in cf.PHYSICS if (c, ph) in reference]
        return math.exp(statistics.mean(math.log(x) for x in ratios)), ratios

    print("\nTRAINING: geometric mean of scheduled time vs old shipped FastSA (lower is better)")
    print(f"{'cost':6s} {'P':>8s} {'thresh':>7s} | {'train':>6s} {'held-out':>8s} | {'mean stage-3 iters':>18s}")
    best = {}
    for cost, p, threshold in itertools.product(cf.COSTS, P_GRID, THRESHOLD_GRID):
        train, _ = geomean_ratio(TRAIN, cost, p, threshold)
        test, _ = geomean_ratio(HELD_OUT, cost, p, threshold)
        print(f"{cost:6s} {p:8.5f} {threshold:7.0e} | {train:6.3f} {test:8.3f} | "
              f"{statistics.mean(stage3[(cost, p, threshold)]):18.0f}")
        if cost not in best or train < best[cost][0]:
            best[cost] = (train, p, threshold)

    print("\nSELECTED on training jobs, reported on held-out jobs")
    for cost, (train, p, threshold) in best.items():
        test, ratios = geomean_ratio(HELD_OUT, cost, p, threshold)
        print(f"  {cost:6s} P={p} threshold={threshold:g}: train {train:.3f}, held-out {test:.3f} "
              f"(range {min(ratios):.3f}-{max(ratios):.3f})")
    (_, pf, tf), (_, pt, tt) = best["fastsa"], best["time"]
    head = [median[(c, ph, "time", pt, tt)] / median[(c, ph, "fastsa", pf, tf)]
            for c in HELD_OUT for ph in cf.PHYSICS]
    every = [median[(c, ph, "time", pt, tt)] / median[(c, ph, "fastsa", pf, tf)]
             for c in (*TRAIN, *HELD_OUT) for ph in cf.PHYSICS]
    for label, ratios in (("held-out", head), ("all jobs", every)):
        geo = math.exp(statistics.mean(math.log(x) for x in ratios))
        print(f"  tuned F_time / tuned F ({label}): geomean {geo:.3f}, "
              f"better {sum(x < 1 for x in ratios)}, worse {sum(x > 1 for x in ratios)}")


if __name__ == "__main__":
    main()
