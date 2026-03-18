"""
find_deg_params.py  —  VECTORIZED EDITION
==========================================
Key change: ALL race data is precomputed into numpy arrays ONCE at startup.
Each loss evaluation then runs entirely in numpy (no Python loops over races).
This is ~50-100x faster than the loop-based version.

Usage:
  python find_deg_params.py --data-dir ../data/historical_races --max-races 30000
"""

import json
import glob
import sys
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

# ==============================================================================
# FIXED KNOWN CONSTANTS
# ==============================================================================

COMPOUND_OFFSET = {"SOFT": -1.0, "MEDIUM": 0.0, "HARD": 0.8}
GRACE_LAPS      = {"SOFT": 10,   "MEDIUM": 20,  "HARD": 30}
T_REF           = 30.0
CI              = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

# ==============================================================================
# PRECOMPUTATION
# ==============================================================================
# For each driver in each race we reduce the entire lap-by-lap simulation
# to a set of coefficients that can be combined with the 4 free parameters
# in a single numpy dot product.
#
# The lap time formula is:
#   lap_time = base + OFFSET[c] + SCALE[c] * base^EXP * laps_past_grace * temp/T_REF
#
# For a full stint the degradation sum is:
#   sum over laps of: SCALE[c] * base^EXP * max(0, age - grace) * temp/T_REF
#
# We can factor out the unknowns:
#   deg_contribution = SCALE[c] * (base^EXP) * (temp/T_REF) * SUM(laps_past_grace)
#
# SUM(laps_past_grace) over a stint of N laps starting at age_start with grace G:
#   = sum_{k=age_start+1}^{age_start+N} max(0, k - G)
#   = analytically computable (triangle number formula)
#
# So each driver record becomes:
#   fixed_time   : base*laps + pit_penalties + sum(OFFSET[c]*stint_laps)  [constant]
#   base_lap_time: needed for base^EXP term                                [per driver]
#   temp_factor  : track_temp / T_REF                                      [per driver]
#   stint_info   : list of (compound_idx, deg_lap_sum)                     [per driver]
#
# Then: total_time = fixed_time + sum_c( SCALE[c] * base^EXP * temp_factor * deg_sum_c )

def _grace_deg_sum(age_start, n_laps, grace):
    """
    Sum of max(0, age - grace) for age in [age_start+1 .. age_start+n_laps].
    Uses triangle number formula — no loop needed.
    """
    if n_laps == 0:
        return 0.0
    # First lap index where degradation kicks in
    first_deg_lap = int(grace) + 1        # first integer age > grace
    lap_start     = age_start + 1
    lap_end       = age_start + n_laps

    if lap_end < first_deg_lap:
        return 0.0                         # entire stint inside grace period

    k_start = max(lap_start, first_deg_lap)
    k_end   = lap_end
    count   = k_end - k_start + 1

    # sum of (k - grace) for k in [k_start, k_end]
    # = sum(k) - grace*count
    # = count*(k_start+k_end)/2 - grace*count
    return count * ((k_start + k_end) / 2.0 - grace)


def precompute_driver(strategy, race_config):
    """
    Returns (fixed_time, base_lap_time, temp_factor, deg_sums[3])
    deg_sums[c] = total degradation sum for compound c across all stints
                  (before multiplying by SCALE[c] * base^EXP)
    """
    base      = race_config["base_lap_time"]
    pit_t     = race_config["pit_lane_time"]
    laps      = race_config["total_laps"]
    temp      = race_config["track_temp"]
    t_factor  = temp / T_REF

    pit_map   = {ps["lap"]: ps["to_tire"] for ps in strategy.get("pit_stops", [])}

    compound    = strategy["starting_tire"]
    stint_start = 1
    age_start   = 0

    fixed_time  = base * laps + pit_t * len(strategy.get("pit_stops", []))
    deg_sums    = np.zeros(3)             # one per compound index

    stints = []
    for lap in range(1, laps + 1):
        if lap in pit_map:
            n = lap - stint_start + 1
            stints.append((CI[compound], age_start, n))
            # Add compound offset contribution
            fixed_time += COMPOUND_OFFSET[compound] * n
            compound    = pit_map[lap]
            age_start   = 0
            stint_start = lap + 1

    # Final stint
    n = laps - stint_start + 1
    if n > 0:
        stints.append((CI[compound], age_start, n))
        fixed_time += COMPOUND_OFFSET[compound] * n

    for cidx, age_start, n_laps in stints:
        grace = list(GRACE_LAPS.values())[cidx]
        deg_sums[cidx] += _grace_deg_sum(age_start, n_laps, grace)

    # Multiply by temp factor now (it's constant per driver)
    deg_sums *= t_factor

    return fixed_time, base, deg_sums


def build_dataset_arrays(dataset):
    """
    Precompute everything into flat numpy arrays.
    Returns arrays ready for vectorized loss calculation.

    For N total drivers across all races:
      fixed_times  : (N,)   constant part of race time
      bases        : (N,)   base_lap_time per driver
      deg_sums     : (N,3)  degradation sums per compound per driver
      pair_faster  : (P,)   index into driver array for the faster driver in each pair
      pair_slower  : (P,)   index into driver array for the slower driver in each pair
    """
    print("  Precomputing race data into numpy arrays...")
    t0 = time.time()

    all_fixed   = []
    all_bases   = []
    all_degsums = []

    pair_faster = []
    pair_slower = []

    driver_offset = 0  # running index into the flat driver arrays

    for race in dataset:
        cfg        = race["race_config"]
        strategies = race["strategies"]
        fp         = race.get("finishing_positions", [])

        # Map driver_id -> local index within this race
        did_to_idx = {}
        for strat in strategies.values():
            did = strat["driver_id"]
            ft, base, ds = precompute_driver(strat, cfg)
            all_fixed.append(ft)
            all_bases.append(base)
            all_degsums.append(ds)
            did_to_idx[did] = driver_offset + len(did_to_idx)

        # Build consecutive pairs from finishing order
        fp_valid = [d for d in fp if d in did_to_idx]
        for i in range(len(fp_valid) - 1):
            pair_faster.append(did_to_idx[fp_valid[i]])
            pair_slower.append(did_to_idx[fp_valid[i + 1]])

        driver_offset += len(did_to_idx)

    fixed_arr   = np.array(all_fixed,   dtype=np.float64)
    bases_arr   = np.array(all_bases,   dtype=np.float64)
    degsums_arr = np.array(all_degsums, dtype=np.float64)  # (N, 3)
    faster_arr  = np.array(pair_faster, dtype=np.int32)
    slower_arr  = np.array(pair_slower, dtype=np.int32)

    elapsed = time.time() - t0
    print(f"  Done. {len(fixed_arr):,} drivers, {len(faster_arr):,} pairs  ({elapsed:.1f}s)\n")

    return fixed_arr, bases_arr, degsums_arr, faster_arr, slower_arr


# ==============================================================================
# VECTORIZED LOSS FUNCTION
# ==============================================================================

_fixed   = None
_bases   = None
_degsums = None
_faster  = None
_slower  = None
_counter = None


def vectorized_loss(params, margin=0.05):
    """
    Entire loss computed with numpy — zero Python loops over races.
    
    params = [scale_soft, scale_medium, scale_hard, exponent]
    
    total_time[i] = fixed[i]
                  + scale_soft   * bases[i]^exp * degsums[i,0]
                  + scale_medium * bases[i]^exp * degsums[i,1]
                  + scale_hard   * bases[i]^exp * degsums[i,2]
    """
    scale_soft, scale_medium, scale_hard, exponent = params

    if scale_soft <= 0 or scale_medium <= 0 or scale_hard <= 0 or exponent < 0:
        if _counter: _counter.update(1e9)
        return 1e9

    scales      = np.array([scale_soft, scale_medium, scale_hard])
    base_pow    = _bases ** exponent                       # (N,)
    deg_contrib = (base_pow[:, None] * _degsums) @ scales # (N,)
    times       = _fixed + deg_contrib                    # (N,)

    t_faster    = times[_faster]                          # (P,)
    t_slower    = times[_slower]                          # (P,)
    violations  = t_faster - t_slower + margin            # (P,)
    loss        = float(np.sum(violations[violations > 0]))

    if _counter: _counter.update(loss)
    return loss


def accuracy_from_arrays(params):
    """Fast accuracy check using the precomputed arrays."""
    scale_soft, scale_medium, scale_hard, exponent = params
    scales      = np.array([scale_soft, scale_medium, scale_hard])
    base_pow    = _bases ** exponent
    deg_contrib = (base_pow[:, None] * _degsums) @ scales
    times       = _fixed + deg_contrib

    # We need to re-group by race to check full ordering
    # Rebuild race groups from the pair arrays
    # Instead, use a quick proxy: fraction of pairs in correct order
    t_faster    = times[_faster]
    t_slower    = times[_slower]
    correct     = float(np.sum(t_faster < t_slower))
    total       = len(_faster)
    return correct / total if total > 0 else 0.0

# ==============================================================================
# PROGRESS COUNTER
# ==============================================================================

class LiveCounter:
    def __init__(self, print_every=10):
        self.calls       = 0
        self.best_loss   = float("inf")
        self.start_time  = time.time()
        self.print_every = print_every

    def update(self, loss):
        self.calls += 1
        if loss < self.best_loss:
            self.best_loss = loss
        if self.calls % self.print_every == 0:
            elapsed = time.time() - self.start_time
            rate    = self.calls / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r  Evals: {self.calls:>6,}  "
                f"Best loss: {self.best_loss:>10.2f}  "
                f"Speed: {rate:>5.0f} evals/s  "
                f"Elapsed: {_fmt_time(elapsed)}   "
            )
            sys.stdout.flush()


def _fmt_time(s):
    s = int(s)
    if s < 60:   return f"{s}s"
    if s < 3600: return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m {s%60:02d}s"

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_dataset(data_dir, max_races=None):
    path = Path(data_dir)
    if not path.is_absolute() and not path.exists():
        script_dir = Path(__file__).resolve().parent
        for c in [script_dir / data_dir, script_dir.parent / data_dir]:
            if c.exists():
                path = c
                break

    files = sorted(glob.glob(str(path / "*.json")))
    if not files:
        print(f"\n  ERROR: No JSON files found in: {path}")
        sys.exit(1)

    dataset = []
    for fidx, fpath in enumerate(files):
        with open(fpath) as f:
            races = json.load(f)
        for r in (races if isinstance(races, list) else [races]):
            if r.get("finishing_positions"):
                dataset.append(r)
                if max_races and len(dataset) >= max_races:
                    print(f"\r  Loaded {len(dataset):,} races.{' '*40}")
                    return dataset
        sys.stdout.write(f"\r  Reading: {fidx+1}/{len(files)} files  |  {len(dataset):,} races...")
        sys.stdout.flush()

    print(f"\r  Loaded {len(dataset):,} races.{' '*50}")
    return dataset

# ==============================================================================
# OPTIMIZER
# ==============================================================================

BOUNDS = [
    (0.001,  0.5),    # scale_soft
    (0.0005, 0.25),   # scale_medium
    (0.0001, 0.15),   # scale_hard
    (0.0,    2.5),    # exponent
]


def run_optimization(pop_size=15, max_iter=300, seed=42):
    global _counter

    baseline_loss = vectorized_loss([1.72, 0.86, 0.43, 0.0])
    print(f"  Baseline loss (original flat rates): {baseline_loss:.2f}\n")

    # ── Phase 1: DE ───────────────────────────────────────────────────
    print(f"{'='*70}")
    print(f"  PHASE 1 — Differential Evolution")
    print(f"  Pop size: {pop_size}   Max iterations: {max_iter}")
    print(f"{'='*70}\n")

    _counter = LiveCounter(print_every=5)

    res_de = differential_evolution(
        func          = vectorized_loss,
        bounds        = BOUNDS,
        seed          = seed,
        popsize       = pop_size,
        maxiter       = max_iter,
        tol           = 1e-8,
        mutation      = (0.5, 1.5),
        recombination = 0.9,
        polish        = False,
        disp          = False,
        workers       = 1,
        updating      = "deferred",
    )

    print(f"\n\n  ✓ DE done.  Loss: {res_de.fun:.4f}  ({_counter.calls:,} evals)")
    print(f"    scale_s={res_de.x[0]:.6f}  scale_m={res_de.x[1]:.6f}  "
          f"scale_h={res_de.x[2]:.6f}  exp={res_de.x[3]:.6f}")

    # ── Phase 2: Nelder-Mead ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  PHASE 2 — Nelder-Mead (local polish)")
    print(f"{'='*70}\n")

    _counter = LiveCounter(print_every=5)

    res_nm = minimize(
        fun     = vectorized_loss,
        x0      = res_de.x,
        method  = "Nelder-Mead",
        options = {"maxiter": 200_000, "xatol": 1e-10, "fatol": 1e-10, "disp": False},
    )

    best      = res_nm.x if res_nm.fun < res_de.fun else res_de.x
    best_loss = min(res_nm.fun, res_de.fun)
    improve   = ((baseline_loss - best_loss) / baseline_loss) * 100

    print(f"\n\n  ✓ Polish done.  Loss: {best_loss:.6f}  ({_counter.calls:,} evals)")
    print(f"  Improvement over baseline: {improve:.1f}%")

    return best

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",  type=str, default="../data/historical_races")
    ap.add_argument("--max-races", type=int, default=30000)
    ap.add_argument("--pop-size",  type=int, default=15)
    ap.add_argument("--max-iter",  type=int, default=300)
    args = ap.parse_args()

    print("=" * 70)
    print("  F1 Degradation Parameter Finder  [VECTORIZED]")
    print("=" * 70)
    print(f"\n  Loading from: {args.data_dir}")

    dataset = load_dataset(args.data_dir, max_races=args.max_races)
    if not dataset:
        print("No races found.")
        sys.exit(1)

    # Precompute everything into numpy arrays
    global _fixed, _bases, _degsums, _faster, _slower
    _fixed, _bases, _degsums, _faster, _slower = build_dataset_arrays(dataset)

    best = run_optimization(pop_size=args.pop_size, max_iter=args.max_iter)
    scale_soft, scale_medium, scale_hard, exponent = best

    # ── Results ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RESULTS")
    print(f"{'='*70}")
    print(f"  scale_soft   = {scale_soft:.8f}")
    print(f"  scale_medium = {scale_medium:.8f}")
    print(f"  scale_hard   = {scale_hard:.8f}")
    print(f"  exponent     = {exponent:.8f}")

    print(f"\n  Compound ratios (ideally ~2.0 each):")
    if scale_medium > 0:
        print(f"    soft / medium  = {scale_soft / scale_medium:.4f}")
    if scale_hard > 0:
        print(f"    medium / hard  = {scale_medium / scale_hard:.4f}")

    print(f"\n  Effective deg rates at common base_lap_times:")
    print(f"  {'lap_time':>10}  {'SOFT':>10}  {'MEDIUM':>10}  {'HARD':>10}")
    print(f"  {'-'*46}")
    for blt in [75.0, 80.0, 85.0, 90.0, 95.0, 100.0]:
        ds = scale_soft   * (blt ** exponent)
        dm = scale_medium * (blt ** exponent)
        dh = scale_hard   * (blt ** exponent)
        print(f"  {blt:>10.1f}s  {ds:>10.5f}  {dm:>10.5f}  {dh:>10.5f}")

    print(f"\n  Pair-level ordering accuracy: {accuracy_from_arrays(best)*100:.1f}%")
    print(f"  (% of consecutive finishing pairs predicted in correct order)")

    # ── Copy-paste block ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  PASTE THIS INTO YOUR RACE SIMULATOR")
    print(f"{'='*70}")
    print(f"""
COMPOUND_OFFSET = {{
    "SOFT":   -1.0,
    "MEDIUM":  0.0,
    "HARD":    0.8,
}}

GRACE_LAPS = {{
    "SOFT":   10,
    "MEDIUM": 20,
    "HARD":   30,
}}

T_REF = 30.0

DEG_SCALE = {{
    "SOFT":   {scale_soft:.8f},
    "MEDIUM": {scale_medium:.8f},
    "HARD":   {scale_hard:.8f},
}}

DEG_EXPONENT = {exponent:.8f}

def calculate_lap_time(base_lap_time, compound, tire_age, track_temp):
    offset          = COMPOUND_OFFSET[compound]
    grace           = GRACE_LAPS[compound]
    laps_past_grace = max(0.0, tire_age - grace)
    temp_factor     = track_temp / T_REF
    deg_rate        = DEG_SCALE[compound] * (base_lap_time ** DEG_EXPONENT)
    degradation     = deg_rate * laps_past_grace * temp_factor
    return base_lap_time + offset + degradation
""")


if __name__ == "__main__":
    main()