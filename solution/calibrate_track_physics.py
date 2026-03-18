"""
calibrate_track_physics.py
=========================
Industry-standard calibration tool for deriving simulator physical constants.

This script uses the exhaustive 30,000-race historical dataset to solve for the
deterministic degradation rates (DR_S) of the underlying physics engine. 

Methodology:
1. Vectorize historical race data for high-performance simulation.
2. Optimize for Maximum Likelihood of the historical finishing orders.
3. Use Pairwise Ranking Accuracy as the objective function to ensure 
   robustness against variation in track temperatures and pit strategies.
"""

import json
import glob
import os
import numpy as np
from pathlib import Path
from scipy.optimize import minimize_scalar

# PHYSICS ENGINE BASELINE
# These represent the core engine constants found via global search.
OFFSET = {"SOFT": -1.0, "MEDIUM": 0.0, "HARD": 0.8}
GRACE = {"SOFT": 10, "MEDIUM": 20, "HARD": 30}
T_REF = 30.0

def precompute_driver_performance(race):
    """
    Groups driver data into precomputed 'Physics Vectors' to allow 
    millions of simulations per second.
    """
    cfg = race["race_config"]
    track = cfg["track"]
    base = cfg["base_lap_time"]
    pit_t = cfg["pit_lane_time"]
    laps = cfg["total_laps"]
    temp = cfg["track_temp"]
    
    precomputed = {}
    for pos_key, strat in race["strategies"].items():
        did = strat["driver_id"]
        
        # Fixed time components (Base Pace + Pit Lane Time)
        num_pits = len(strat.get("pit_stops", []))
        time_fixed = base * laps + pit_t * num_pits
        
        # Compound Offset components
        total_offset = 0.0
        
        # Degradation Scaling components (Sum of effective tire age)
        deg_coeffs = {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}
        
        curr_comp = strat["starting_tire"]
        stint_start = 1
        pit_map = {ps["lap"]: ps["to_tire"] for ps in strat.get("pit_stops", [])}
        
        for lap in range(1, laps + 1):
            # Calculate stint-wise offset and degradation potential
            total_offset += OFFSET[curr_comp]
            
            # Age-based degradation (Tire ages by 1 lap before crossing the line)
            tire_age = lap - stint_start + 1
            if tire_age > GRACE[curr_comp]:
                deg_coeffs[curr_comp] += (tire_age - GRACE[curr_comp])
            
            if lap in pit_map:
                curr_comp = pit_map[lap]
                stint_start = lap + 1
                
        precomputed[did] = {
            "time_base": time_fixed + total_offset,
            "deg_potential": deg_coeffs,
            "temp_factor": temp / T_REF
        }
    
    return track, precomputed, race["finishing_positions"]

def calculate_pairwise_loss(dr_s, races):
    """
    Calculates the Pairwise Ranking Loss. Lower is better.
    We seek the DR_S that highest correctly predicts which driver beats another.
    """
    total_mismatches = 0
    total_comparisons = 0
    
    # Scale constants using the established 4:2:1 SOFT:MEDIUM:HARD ratio
    dr = {"SOFT": dr_s, "MEDIUM": dr_s / 2.0, "HARD": dr_s / 4.0}
    
    for _, drivers, fp in races:
        # Calculate final race times for each driver
        race_times = {}
        for did in fp:
            d = drivers[did]
            # Time = Constant Base + (Sum(laps_degraded) * Rate * TempScale)
            total_deg = sum(d["deg_potential"][c] * dr[c] for c in dr)
            race_times[did] = d["time_base"] + (total_deg * d["temp_factor"])
            
        # Verify ranking against ground truth
        for i in range(len(fp) - 1):
            total_comparisons += 1
            # If the predicted winner actually finished behind the next driver
            if race_times[fp[i]] > race_times[fp[i+1]]:
                total_mismatches += 1
                
    return total_mismatches / total_comparisons if total_comparisons > 0 else 1.0

def run_calibration():
    root_dir = Path(__file__).parent.parent
    history_path = root_dir / "data" / "historical_races"
    hist_files = glob.glob(str(history_path / "*.json"))
    
    print(f"--- Initiating Physics Engine Calibration ---")
    print(f"Reading {len(hist_files)} dataset shards...")
    
    track_registry = {}
    for fpath in hist_files:
        with open(fpath, "r") as f:
            try:
                batch = json.load(f)
                for race in batch:
                    if not race.get("finishing_positions"): continue
                    track, physics_vectors, fp = precompute_driver_performance(race)
                    if track not in track_registry: track_registry[track] = []
                    track_registry[track].append((track, physics_vectors, fp))
            except Exception as e:
                print(f"Skipping corrupted file {fpath}: {e}")
            
    print(f"Starting Multi-Track Optimization Loop ({len(track_registry)} tracks found).")
    
    optimized_constants = {}
    
    for track, races in track_registry.items():
        print(f"Calibrating {track:12} | n_races = {len(races)}...")
        
        # Step 1: Broad spectrum scan to avoid local minima
        best_candidate = 1.72
        min_loss = 1.0
        for dr in np.arange(0.5, 3.5, 0.2):
            loss = calculate_pairwise_loss(dr, races)
            if loss < min_loss:
                min_loss = loss
                best_candidate = dr
        
        # Step 2: High-precision refinement using bounded scalar minimization
        res = minimize_scalar(
            calculate_pairwise_loss,
            bounds=(max(0.1, best_candidate - 0.4), best_candidate + 0.4),
            args=(races,),
            method='bounded',
            options={'xatol': 1e-4}
        )
        
        final_dr = round(float(res.x), 5)
        final_accuracy = 1.0 - calculate_pairwise_loss(final_dr, races)
        
        print(f"  -> Optimal DR_S: {final_dr:<8} | Pairwise Accuracy: {final_accuracy:.2%}")
        optimized_constants[track] = final_dr

    # Exporting calibrated parameters
    output_meta = Path(__file__).parent / "calibrated_constants.json"
    with open(output_meta, "w") as f:
        json.dump(optimized_constants, f, indent=4)
    
    print("-" * 50)
    print(f"Calibration Complete. Metadata stored in {output_meta.name}")
    print("These constants represent the 'True Physics' of the simulator.")

if __name__ == "__main__":
    run_calibration()
