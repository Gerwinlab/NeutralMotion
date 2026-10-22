#!/usr/bin/env python3
"""Find the fastest certified block order for the P126 / P254 syndrome rounds.

The 12 CZ blocks of a P126 or P254 syndrome-extraction round can run in many
orders, but only the orders certified in ``../syndrome_qasm/sets/<code>.json``
are known to keep the code's full circuit distance (714 for P126, 4 for P254).
This script exports each certified order with ``syndrome_qasm/export_qasm.py``
(its classical registers ``sx``/``sz`` renamed to ``c_x``/``c_z``, since ``sx``
is a qelib1.inc gate), places it with the deterministic placer and schedules
it, and ranks the orders by scheduled time.

The deterministic placer's ``budget`` sets the cost of each order; a small
budget screens all orders quickly, and ``--ranks`` re-runs a shortlist at a
larger budget. Results go to ``benchmarks/output/results/order_search_<code>
_b<budget>.csv``; ``--save-best`` also writes the fastest order's QASM to
``inputs/qasm_files/qLDPC/<code>_rank<rank>.qasm``.

Usage (from the repo root):

    ~/Quantum_Class/Phys765_Test/bin/python benchmarks/order_search.py P126 --budget 300 --workers 9
    ~/Quantum_Class/Phys765_Test/bin/python benchmarks/order_search.py P126 --budget 20000 \\
        --ranks 188399054 444068163 --save-best
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import multiprocessing
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "src"))
DEFAULT_SYNDROME_DIR = PROJECT.parent / "syndrome_qasm"
REFERENCE_RANK = {"P126": 188399054, "P254": 179361634}


def load_exporter(syndrome_dir: Path):
    spec = importlib.util.spec_from_file_location("syndrome_export_qasm", syndrome_dir / "export_qasm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qasm_for(exporter, code: str, entry) -> str:
    """The exporter's QASM with the syndrome registers renamed off qelib1 gate names."""
    text = exporter.generate_qasm(code, entry)
    return re.sub(r"\bs([xz])\b", r"c_\1", text)


def evaluate(job):
    """Place and schedule one order; returns its rank, order, time (us) and wall seconds."""
    code, entry, qasm_text, config_path, budget, work_dir = job
    warnings.simplefilter("ignore")
    from naive_n_dag.main import load_config, main

    config = load_config(Path(config_path))
    config.update(fill_strategy="deterministic", Deterministic_fill={"budget": budget})
    config.pop("Fastsa_fill", None)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        qasm = Path(tmp) / f"{code}_rank{entry['rank']}.qasm"
        qasm.write_text(qasm_text)
        config["qasm_base_dir"] = tmp
        start = time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()):
            main(config, qasm.name, schedule_output_dir=tmp, quiet=True)
        wall = time.monotonic() - start
        text = next(Path(tmp).glob("*.schedule.txt")).read_text()
    final = float(re.search(r"final_time:\s*([0-9.eE+-]+)", text).group(1))
    return entry["rank"], entry["order"], final, wall


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("code", type=str.upper, choices=sorted(REFERENCE_RANK))
    parser.add_argument("--config", type=Path, default=None,
                        help="Scheduler config (default: benchmarks/configs/qLDPC/<code>_rank<ref>_deterministic.json)")
    parser.add_argument("--budget", type=int, default=300, help="Deterministic placer budget per order")
    parser.add_argument("--ranks", type=int, nargs="+", help="Only these certified ranks")
    parser.add_argument("--limit", type=int, help="Only the first N certified orders (quick tests)")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--syndrome-dir", type=Path, default=DEFAULT_SYNDROME_DIR)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--save-best", action="store_true", help="Write the fastest order's QASM to inputs/")
    args = parser.parse_args()

    exporter = load_exporter(args.syndrome_dir.resolve())
    entries = exporter.load_set(args.code)["schedules"]
    if args.ranks:
        missing = set(args.ranks) - {e["rank"] for e in entries}
        if missing:
            parser.error(f"Not certified {args.code} ranks: {sorted(missing)}")
        entries = [e for e in entries if e["rank"] in set(args.ranks)]
    if args.limit:
        entries = entries[:args.limit]
    config = args.config or (HERE / "configs/qLDPC" / f"{args.code}_rank{REFERENCE_RANK[args.code]}_deterministic.json")
    csv_path = args.csv or HERE / "output/results" / f"order_search_{args.code}_b{args.budget}.csv"
    work_dir = PROJECT / "benchmarks" / f".order_search_{args.code}"
    work_dir.mkdir(exist_ok=True)
    jobs = [(args.code, e, qasm_for(exporter, args.code, e), str(config.resolve()), args.budget, str(work_dir))
            for e in entries]
    print(f"{args.code}: {len(jobs)} certified orders, budget {args.budget}, {args.workers} workers, config {config.name}",
          flush=True)

    results, start = [], time.monotonic()
    try:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for k, result in enumerate(pool.imap_unordered(evaluate, jobs), 1):
                results.append(result)
                best = min(results, key=lambda r: r[2])
                if k % max(1, len(jobs) // 20) == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} done ({time.monotonic() - start:.0f} s); "
                          f"best so far rank {best[0]}: {best[2]:.1f} us", flush=True)
    finally:
        for leftover in work_dir.glob("*"):
            leftover.rmdir() if leftover.is_dir() else leftover.unlink()
        work_dir.rmdir()

    results.sort(key=lambda r: r[2])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "order", "final_time_us", "wall_s", "budget"])
        for rank, order, final, wall in results:
            writer.writerow([rank, " ".join(map(str, order)), f"{final:.4f}", f"{wall:.2f}", args.budget])
    reference = next((r for r in results if r[0] == REFERENCE_RANK[args.code]), None)
    print(f"\nFastest orders (of {len(results)}):")
    for rank, order, final, wall in results[:10]:
        vs = f"  ({final / reference[2] - 1:+.1%} vs reference)" if reference else ""
        print(f"  rank {rank:>10d}  order {' '.join(f'{o:2d}' for o in order)}  {final:9.1f} us{vs}")
    if reference:
        print(f"  reference rank {reference[0]}: {reference[2]:.1f} us")
    print(f"Wrote {csv_path}")
    if args.save_best:
        rank = results[0][0]
        entry = next(e for e in entries if e["rank"] == rank)
        out = PROJECT / "inputs/qasm_files/qLDPC" / f"{args.code}_rank{rank}.qasm"
        out.write_text(qasm_for(exporter, args.code, entry))
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
