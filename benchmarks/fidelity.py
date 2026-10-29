"""Fidelity estimate of a schedule file, after the Atomique model.

F = F_1Q * F_2Q * F_transfer * F_mov, with F_mov = heating * motion loss *
cooling * decoherence (H. Wang et al., "Atomique: A Quantum Compiler for
Reconfigurable Neutral Atom Arrays", ISCA 2024, Sec. IV and V-A; parameters
from its Table I). Everything is read from the schedule: gates and transfers
are counted from its statements, decoherence uses its final_time, and each
straight move leg heats its atoms by the formula for its movement_shape. The
config only supplies max_acceleration, movement_shape and (for Quintic_Spline)
lattice_spacing. Details: benchmarks/FIDELITY.md.

Usage (from the repository root):

    python benchmarks/fidelity.py <schedule.txt> <config.json> [--csv trace.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re

import pint

# Atomique Table I (neutral atoms).
F_1Q = 0.99992            # one-qubit gate fidelity
F_2Q = 0.9975             # two-qubit gate fidelity
P_LOSS_TRANSFER = 0.0068  # atom loss per SLM <-> AOD transfer
T1 = 15.0                 # coherence time, s
X_ZPF = 38e-9             # zero-point size of the trapped atom, m
OMEGA_0 = 2 * math.pi * 80e3  # trap frequency, rad/s
LAMBDA = 0.109            # heating error per vibrational quantum, in units of 2Q error
N_VIB_MAX = 33            # n_vib scale of the motion-induced loss formula
N_VIB_COOL = 15           # cooling threshold
QUINTIC = 10 / math.sqrt(3)  # a quintic leg of length D with peak acceleration a takes sqrt(QUINTIC D / a)

_STEP = re.compile(r"^T=(\d+)$")
_FINAL = re.compile(r"^final_time:\s*([0-9eE+\-.]+)\s*microsecond")
_INIT = re.compile(r"q\[(\d+)\] -> \((-?\d+),(-?\d+)\)")
_MOVE = re.compile(r"(load|unload|move) q\[(\d+)\](?: ->)? \((-?\d+),(-?\d+)\) : \((-?\d+),(-?\d+)\)")
_QUBIT = re.compile(r"q\[(\d+)\]")


def delta_n_vib(distance, seconds):
    """Vibrational quanta added by one constant-jerk move of ``distance`` m in ``seconds``."""
    if distance == 0 or seconds == 0:
        return 0.0
    return 0.5 * (6 * distance / (X_ZPF * OMEGA_0**2 * seconds**2)) ** 2


def leg_n_vib(acceleration):
    """Quanta per straight leg of a cubic move with peak acceleration ``acceleration`` (6D/T^2 = a)."""
    return 0.5 * (acceleration / (X_ZPF * OMEGA_0**2)) ** 2


def quintic_leg_n_vib(distance, acceleration):
    """Quanta per minimum-jerk leg of ``distance`` m with peak acceleration ``acceleration``.

    Phase-averaged |FT of a(t) at omega_0|^2 / (4 x_zpf^2 omega_0^2), whose leading
    term for the cubic is leg_n_vib. The quintic starts and ends at zero acceleration,
    so its leading term comes from the jerk jumps and depends on the length.
    """
    if distance == 0:
        return 0.0
    seconds = math.sqrt(QUINTIC * distance / acceleration)
    wt = OMEGA_0 * seconds
    per_end = (distance / (seconds**3 * OMEGA_0**2)) ** 2 * ((60 - 720 / wt**2) ** 2 + (360 / wt) ** 2)
    return 2 * per_end / (4 * X_ZPF**2 * OMEGA_0**2)


def p_loss_motion(n_vib):
    """Probability that an atom with ``n_vib`` quanta is lost when moved."""
    if n_vib <= 0:
        return 0.0
    return 1 - 0.5 * (1 + math.erf((N_VIB_MAX - n_vib) / math.sqrt(2 * n_vib)))


def _slm(position):
    return position[0] % 2 == 0 and position[1] % 2 == 0


def estimate(schedule_path, config):
    """Return (factors, counts, rows, n_atoms, final_time_us) for one schedule."""
    units = pint.UnitRegistry()
    acceleration = units.Quantity(config["max_acceleration"]).to("m/s^2").magnitude
    per_leg = leg_n_vib(acceleration)
    shape = config.get("movement_shape")
    if shape == "Quintic_Spline":
        grid_step = units.Quantity(config["lattice_spacing"]).to("m").magnitude / 2

    def heat(start, end):
        """(legs, quanta) of one straight move statement; spline shapes move the axes in turn."""
        cells = [abs(end[0] - start[0]), abs(end[1] - start[1])]
        legs = sum(map(bool, cells)) if shape in ("Cubic_Spline", "Quintic_Spline") else 1
        if shape == "Quintic_Spline":
            return legs, sum(quintic_leg_n_vib(c * grid_step, acceleration) for c in cells)
        return legs, legs * per_leg

    factors = dict.fromkeys(("1q gates", "2q gates", "heating", "transfer loss", "motion loss", "cooling"), 1.0)
    counts = dict.fromkeys(("1q gates", "2q gates", "transfers", "moved legs", "cooling events"), 0)
    n_vib, loaded, rows = {}, set(), []
    final_time_us, step = None, None
    for raw in pathlib.Path(schedule_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if _FINAL.match(line):
            final_time_us = float(_FINAL.match(line).group(1))
        if _STEP.match(line):
            step = int(_STEP.match(line).group(1))
            continue
        if step is None or not line or line.startswith("//"):
            continue
        if step == 0:
            n_vib = {int(q): 0.0 for q, _, _ in _INIT.findall(line)}
            continue
        moves = _MOVE.findall(line)
        if moves:
            for kind, q, r1, c1, r2, c2 in moves:
                q, start, end = int(q), (int(r1), int(c1)), (int(r2), int(c2))
                legs, quanta = heat(start, end)
                n_vib[q] += quanta
                counts["moved legs"] += legs
                factors["motion loss"] *= 1 - p_loss_motion(n_vib[q])
                if kind in ("load", "unload"):
                    factors["transfer loss"] *= 1 - P_LOSS_TRANSFER
                    counts["transfers"] += 1
                    (loaded.add if kind == "load" else loaded.discard)(q)
            # Cooling swaps the whole AOD with a cooled array: two CZs per AOD atom, quanta reset.
            if any(n_vib[q] > N_VIB_COOL for q in loaded):
                factors["cooling"] *= F_2Q ** (2 * len(loaded))
                counts["cooling events"] += 1
                for q in loaded:
                    n_vib[q] = 0.0
            kind = "move"
        else:
            for statement in filter(None, (s.strip() for s in line.split(";"))):
                if statement.startswith(("measure", "barrier")):
                    continue
                ids = [int(q) for q in _QUBIT.findall(statement.split("->")[0])]
                if len(ids) == 1:
                    factors["1q gates"] *= F_1Q
                    counts["1q gates"] += 1
                elif len(ids) == 2:
                    factors["2q gates"] *= F_2Q
                    counts["2q gates"] += 1
                    # Heating counts the AOD atom's quanta (summed if both are in the AOD).
                    hot = sum(n_vib[q] for q in ids if q in loaded)
                    factors["heating"] *= 1 - LAMBDA * (1 - F_2Q) * hot
            kind = "gate"
        rows.append({"step": step, "kind": kind, "max n_vib": max(n_vib.values(), default=0.0),
                     **{f"F {k}": v for k, v in factors.items()}, "F (no decoherence)": math.prod(factors.values())})
    if final_time_us is None:
        raise ValueError(f"{schedule_path}: no final_time in the header")
    return factors, counts, rows, len(n_vib), final_time_us


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("schedule", type=pathlib.Path, help="schedule .txt written by naive_n_dag or naive_dag")
    parser.add_argument("config", type=pathlib.Path, help="the JSON config the schedule was made with")
    parser.add_argument("--csv", type=pathlib.Path, help="write the per-timestep trace to this CSV")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    factors, counts, rows, n_atoms, final_time_us = estimate(args.schedule, config)
    # Decoherence of every atom over the whole schedule: exp(-N * T / T1).
    factors["decoherence"] = math.exp(-n_atoms * final_time_us * 1e-6 / T1)
    print(f"atoms={n_atoms} timesteps={len(rows)} final_time={final_time_us:.6g} us")
    print("counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print("factors: " + ", ".join(f"{k}={v:.6g}" for k, v in factors.items()))
    print(f"fidelity={math.prod(factors.values()):.6g}")
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"trace={args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
