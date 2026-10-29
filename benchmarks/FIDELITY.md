# Schedule fidelity (`fidelity.py`)

`fidelity.py` estimates the fidelity of a schedule file written by `naive_n_dag`
or `naive_dag`, using the fidelity model of Atomique:

> H. Wang, P. Liu, D. B. Tan, Y. Liu, J. Gu, D. Z. Pan, J. Cong, U. A. Acar,
> S. Han, "Atomique: A Quantum Compiler for Reconfigurable Neutral Atom
> Arrays", ISCA 2024 (Sec. IV and V-A, Table I).

```bash
python benchmarks/fidelity.py <schedule.txt> <config.json> [--csv trace.csv]
```

It prints the atom and timestep counts, the operation counts, every fidelity
factor and the total. `--csv` writes a per-timestep trace: the cumulative
factors after each step, the largest vibrational number, and the running
fidelity without decoherence (decoherence is applied once, over the whole
schedule).

Everything is read from the schedule: gates and transfers are counted from its
statements, and decoherence uses its `final_time`. Nothing is re-timed. The
config only supplies `max_acceleration`, `movement_shape` and, for
`Quintic_Spline`, `lattice_spacing`.

## Model

```text
F = F_1Q * F_2Q * F_transfer * F_mov
F_mov = F_heating * F_motion_loss * F_cooling * F_decoherence
```

| Factor | How it is computed from the schedule |
|---|---|
| `F_1Q` | `f_1Q ^ N_1Q`, one factor per one-qubit statement (resets included, measurements excluded) |
| `F_2Q` | `f_2Q ^ N_2Q`, one factor per two-qubit statement |
| `F_heating` | per two-qubit gate, `1 - lambda * (1 - f_2Q) * n_vib`, where `n_vib` is the vibrational number of the operand in the AOD (summed if both are) |
| `F_transfer` | `(1 - P_loss) ^ N_transfer`, one factor per atom on a `load` or `unload` line |
| `F_motion_loss` | per moved atom and step, `1 - P_mov_loss(n_vib)` with `P_mov_loss = 1 - (1 + erf((n_max - n_vib) / sqrt(2 n_vib))) / 2` |
| `F_cooling` | when an AOD atom's `n_vib` exceeds the threshold, the whole AOD is swapped with a cooled array: `f_2Q ^ (2 N_AOD)`, and the AOD atoms' `n_vib` reset to 0 |
| `F_decoherence` | `exp(-N * T / T1)` with `N` atoms and `T` the schedule's `final_time` |

### Motion heating

Atomique's heating per move with the constant-jerk profile is

```text
delta_n_vib = 1/2 * (6 D / (x_zpf * omega_0^2 * T_mov^2))^2
```

for a move of distance `D` in time `T_mov`. Atomique fixes `T_mov = 300 us`.
Our `Cubic_Spline` motion is that same profile, but timed by the acceleration
limit: a straight leg of length `D` takes `T = sqrt(6 D / a)`, so `6 D / T^2 = a`
and every leg adds the same

```text
delta_n_vib = 1/2 * (a / (x_zpf * omega_0^2))^2
```

whatever its length. Each `move`/`load`/`unload` statement in the schedule is
one straight segment; with `Cubic_Spline` its row and column legs run in turn,
so it counts one leg per axis it moves along. Held atoms (printed as `move`
lines that touch an SLM site during a transfer step) heat like any other moved
atom but are not transfers. For `trapezoidal` configs the constant-jerk formula
does not strictly apply; one leg per statement is counted as an estimate.
`Quintic_Spline` also counts one leg per axis, each heated by its own length
(see below).

| `max_acceleration` | quanta per leg | legs before cooling |
|---|---|---|
| 2750 m/s^2 (PowerMove settings) | 0.041 | about 366 |
| 20000 m/s^2 (qLDPC configs) | 2.17 | about 7 |

The general result for an atom in a harmonic trap whose centre follows a
rest-to-rest acceleration `a(t)` is

```text
delta_n_vib = |integral a(t) exp(i omega_0 t) dt|^2 / (4 x_zpf^2 omega_0^2)
```

Integrating by parts, only the jumps of `a(t)` and its derivatives at the two
ends contribute. Averaged over the phase `omega_0 T`, the cubic's jumps in
acceleration give Atomique's formula above (to within `1 + 4/(omega_0 T)^2`).
The exact value for one leg swings between 0 and twice the average with
`omega_0 T`; since trap frequencies differ slightly between tweezers and
transfers scramble the phase, the average is used.

### Minimum-jerk (`Quintic_Spline`) moves

`Quintic_Spline` moves along `x = D (10 tau^3 - 15 tau^4 + 6 tau^5)`,
`tau = t / T`. Its acceleration starts and ends at zero and peaks at
`10 / sqrt(3) * D / T^2`, so a leg limited to peak acceleration `a` takes
`T = sqrt(10 D / (sqrt(3) a))`, about 2% less than the cubic. Only the jumps in
jerk at the ends remain, and the same phase-averaged integral gives

```text
delta_n_vib = 2 (D / (T^3 omega_0^2))^2 ((60 - 720/(omega_0 T)^2)^2 + (360/(omega_0 T))^2)
              / (4 x_zpf^2 omega_0^2)
```

which falls roughly as `1/D`: longer legs heat less. Quanta per leg with the
Atomique constants (one grid cell = 7.5 um):

| `max_acceleration` | 1 cell | 2 cells | 10 cells | 20 cells | cubic, any length |
|---|---|---|---|---|---|
| 2750 m/s^2 | 1.1e-3 | 5.6e-4 | 1.1e-4 | 5.6e-5 | 0.041 |
| 20000 m/s^2 | 0.44 | 0.22 | 0.043 | 0.021 | 2.17 |

For BV-50 at 2750 m/s^2 (`naive_n_dag`, deterministic placement) the schedule
shortens from 7.40 to 7.26 ms and the heating factor rises from 0.9958 to
0.99994; the fidelity goes from 0.904 to 0.908. `tests/test_fidelity_script.py`
checks the formula against a direct numerical integral of the profile.

**Experimental use.** The minimum-jerk profile was introduced for arm
movements:

> T. Flash and N. Hogan, "The coordination of arm movements: an experimentally
> confirmed mathematical model", J. Neurosci. 5, 1688 (1985).

Tweezer experiments use it to move single atoms without motional excitation:

> L. R. Liu, J. D. Hood, Y. Yu, J. T. Zhang, K. Wang, Y.-W. Lin, T. Rosenband,
> K.-K. Ni, "Molecular assembly of ground-state cooled single atoms", Phys.
> Rev. X 9, 021039 (2019), Appendix A.

They move a ground-state-cooled Cs atom by sweeping the AOD along
`x = d (10 tau^3 - 15 tau^4 + 6 tau^5)`, chosen to avoid "heating due to jerk
(time-derivative of acceleration) at the endpoints". A 2.5 um move in 7.6 ms
and its time reverse added 0.01(5) axial and 0.00(2) radial quanta (Fig. 3b).

> S. Spence, R. V. Brooks, D. K. Ruttley, A. Guttridge, S. L. Cornish,
> "Preparation of 87Rb and 133Cs in the motional ground state of a single
> optical tweezer", New J. Phys. 24, 103022 (2022), Sec. 6 and Appendix E.

They note that "the sudden change in acceleration at the start and end of the
sweep can cause motional excitation" and that a minimum-jerk trajectory
"resolves the heating associated with jerk".

Both groups then use a hybrid: minimum-jerk ends with a constant-speed middle
(95% of the 2.95 um merges of Fig. 3c in Liu et al.; 10% of a 4.5 um,
1.6 ms move in Spence et al.). A pure minimum-jerk move sweeps the AOD drive
frequency at a varying rate, and AOD intensity ripple at that rate can
parametrically heat the atom when it matches a trap resonance. That
parametric heating is not modelled here. These moves are also much slower
than ours (`omega_0 T` of several hundred or more, against about 90 for a
15 um leg at 2750 m/s^2), so they show the profile in use, not its heating at our speeds.

## Parameters (Atomique Table I, hard-coded)

| Constant | Value | Meaning |
|---|---|---|
| `F_1Q` | 0.99992 | one-qubit gate fidelity |
| `F_2Q` | 0.9975 | two-qubit gate fidelity |
| `P_LOSS_TRANSFER` | 0.0068 | atom loss per SLM-AOD transfer |
| `T1` | 15 s | coherence time |
| `X_ZPF` | 38 nm | zero-point size of the trapped atom |
| `OMEGA_0` | 2 pi x 80 kHz | trap frequency |
| `LAMBDA` | 0.109 | heating error per quantum, in units of the 2Q error |
| `N_VIB_MAX` | 33 | `n_vib` scale of the motion-loss formula |
| `N_VIB_COOL` | 15 | cooling threshold |

The formulas reproduce the paper's worked numbers: 0.0054, 0.13 and 0.54 quanta
for 300 us moves of 1, 5 and 10 hops of 15 um, and motion-loss fidelities of
0.708, 0.998 and 0.999998 at `n_vib` = 30, 20 and 15
(`tests/test_fidelity_script.py`).

## Differences from Atomique and open choices

- **Table I versus the text.** Table I lists `f_1Q = 0.99992` and `T1 = 15 s`;
  the text quotes 0.9992 and 1.5 s for current hardware. Table I is the scaled
  setting the paper evaluates with, and is used here.
- **Move time.** Atomique charges a fixed 300 us per move; here each leg's time
  follows from the configured acceleration, which makes the heating per leg
  constant.
- **Cooling** is bookkeeping only: our schedules contain no cooling operation
  and only one AOD, so the cooling factor and reset model what a second,
  pre-cooled AOD array would cost.
- **Decoherence** uses the whole `final_time`, which also includes `t_switch`
  and measurement time; Atomique sums gate, transfer and move times only.
- **Zero-point size.** Atomique's `x_zpf = 38 nm` is `sqrt(hbar / (m omega_0))`
  for Rb-87 at 80 kHz. The usual definition `sqrt(hbar / (2 m omega_0))` gives
  27 nm, with which the phase-averaged integral for the cubic is twice
  Atomique's formula. The published 38 nm is kept, for both splines, so the
  numbers stay comparable with the paper.
- **Heating at the gate** uses only the AOD operand's `n_vib`, as in Atomique;
  an SLM atom that was moved earlier keeps its quanta but does not add to the
  gate error until it is loaded again.
