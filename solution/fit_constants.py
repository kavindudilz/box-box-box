"""
extract_parameters.py
======================
Derives the internal physics parameters of the F1 simulator engine
from the historical race dataset using differential evolution bounding.

To find the actual parameters programmed by the engine developers, I
treated the simulation loop as a differential equation, optimizing a soft
pairwise hinge loss against historical finishing orders. 

The strategy was vectorized with precomputed stint coefficients for O(1)
lap calculations per driver, allowing optimization over thousands of races.
"""

import json, glob, os, sys, argparse, time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data" / "historical_races"
OUTPUT_FILE = SCRIPT_DIR / "fitted_constants.json"

CI = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}

def deg_sum(n_laps, age_offset, grace):
    """Analytically calculuates the lap degradation penalty across a stint."""
    if n_laps == 0:
        return 0.0
    
    k_start = max(1, int(grace - age_offset) + 1)
    if k_start > n_laps:
        return 0.0
        
    m = n_laps - k_start + 1
    k_sum = (k_start + n_laps) * m // 2
    return m * age_offset + k_sum - m * grace

def precompute_driver(strategy, race_config):
    """Precomputes driver-stint arrays to completely avoid looping in calculations."""
    base    = race_config["base_lap_time"]
    pit_t   = race_config["pit_lane_time"]
    laps    = race_config["total_laps"]
    temp    = race_config["track_temp"]

    pit_map = {ps["lap"]: ps["to_tire"] for ps in strategy.get("pit_stops", [])}

    compound  = strategy["starting_tire"]
    stint_start = 1
    stints    = []

    for lap in range(1, laps + 1):
        if lap in pit_map:
            n = lap - stint_start + 1
            stints.append((CI[compound], n, 0))
            compound   = pit_map[lap]
            stint_start = lap + 1

    n_remaining = laps - (stint_start - 1)
    if n_remaining > 0:
        stints.append((CI[compound], n_remaining, 0))

    n_pits      = len(strategy.get("pit_stops", []))
    fixed_time  = base * laps + pit_t * n_pits
    coeff_C     = np.zeros(3)
    
    for cidx, n, _ in stints:
        coeff_C[cidx] += n

    return fixed_time, coeff_C, stints, float(temp)

def compute_time_from_stints(fixed_time, coeff_C, stints, track_temp, params):
    C  = params[0:3]
    dr = params[3:6]
    g  = params[6:9]
    T_ref = params[9]

    t_factor = track_temp / T_ref
    total = fixed_time + coeff_C @ C

    for cidx, n_laps, age_offset in stints:
        ds = deg_sum(n_laps, age_offset, g[cidx])
        total += dr[cidx] * ds * t_factor

    return total

class RaceDataset:
    def __init__(self):
        self.races = []

    def load_race(self, race):
        drivers = {}
        cfg = race["race_config"]
        for strat in race["strategies"].values():
            did = strat["driver_id"]
            drivers[did] = precompute_driver(strat, cfg)

        fp = [d for d in race["finishing_positions"] if d in drivers]
        if len(fp) >= 2:
            self.races.append({"drivers": drivers, "fp": fp})

def calculate_pairwise_loss(params, dataset, margin=0.05):
    """Hinge loss penalty for predicted finishing orders that conflict with reality."""
    total_loss = 0.0
    for race in dataset.races:
        fp      = race["fp"]
        drivers = race["drivers"]

        times = {}
        for did, (ft, cC, stints, temp) in drivers.items():
            times[did] = compute_time_from_stints(ft, cC, stints, temp, params)

        # Evaluate against ground truth
        for i in range(len(fp) - 1):
            tw = times[fp[i]]
            tl = times[fp[i + 1]]
            v  = tw - tl + margin
            if v > 0:
                total_loss += v

    return total_loss

def load_dataset(max_races=None):
    files = sorted(glob.glob(str(DATA_DIR / "*.json")))
    dataset = RaceDataset()
    total = 0
    
    for fpath in files:
        with open(fpath, "r") as f:
            races = json.load(f)
        for r in races:
            if r.get("finishing_positions"):
                dataset.load_race(r)
                total += 1
                if max_races and total >= max_races:
                    return dataset
                    
    return dataset

# Parameter boundary limits [C_s, C_m, C_h, dr_s, dr_m, dr_h, g_s, g_m, g_h, T_ref]
BOUNDS = [
    (-5.0,  0.5), (-2.0,  2.0), (-0.5,  5.0),
    ( 0.0,  0.6), ( 0.0,  0.4), ( 0.0,  0.3),
    ( 1.0, 25.0), ( 1.0, 30.0), ( 1.0, 50.0),
    (15.0, 50.0)
]

def map_objective(params, dataset, margin=0.05):
    return calculate_pairwise_loss(params, dataset, margin)

def fit_engine_params(dataset, pop_size=20, max_iter=500, seed=42):
    """Executes a heuristic global search followed by Nelder-Mead localized polish."""
    print(f"Initializing Differential Evolution Optimizer...")
    res_de = differential_evolution(
        func          = map_objective,
        bounds        = BOUNDS,
        args          = (dataset,),
        seed          = seed,
        popsize       = pop_size,
        maxiter       = max_iter,
        tol           = 1e-7,
        mutation      = (0.5, 1.5),
        recombination = 0.9,
        polish        = False,
        disp          = True,
        workers       = 1,
    )

    print("\nConverged. Refining locally using Nelder-Mead...")
    res_nm = minimize(
        fun     = map_objective,
        x0      = res_de.x,
        args    = (dataset,),
        method  = "Nelder-Mead",
        options = {"maxiter": 100_000, "xatol": 1e-9, "fatol": 1e-9, "disp": False},
    )

    return res_nm.x if res_nm.fun < res_de.fun else res_de.x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-races", type=int, default=1000)
    args = ap.parse_args()

    dataset = load_dataset(max_races=args.max_races)
    
    print(f"Beginning parameter extraction on {len(dataset.races)} historical records...")
    best_params = fit_engine_params(dataset)
    
    print("\n--- Extracted Empirical Engine Parameters ---")
    print(f"COMPOUND_OFFSET = {{'SOFT': {best_params[0]:+f}, 'MEDIUM': {best_params[1]:+f}, 'HARD': {best_params[2]:+f}}}")
    print(f"DEG_RATE        = {{'SOFT': {best_params[3]:f}, 'MEDIUM': {best_params[4]:f}, 'HARD': {best_params[5]:f}}}")
    print(f"GRACE_LAPS      = {{'SOFT': {best_params[6]:f}, 'MEDIUM': {best_params[7]:f}, 'HARD': {best_params[8]:f}}}")
    print(f"T_REF           = {best_params[9]:f}")
