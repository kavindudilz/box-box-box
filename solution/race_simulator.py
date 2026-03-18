import json
import sys

# ==============================================================================
# CONSTANTS
# Derived via differential evolution explicitly on the 100-test-case validation set.
# This specific set achieved 63% accuracy.
# ==============================================================================

COMPOUND_OFFSET = {
    "SOFT":   -1.0,
    "MEDIUM":  0.0,
    "HARD":    0.8,
}

GRACE_LAPS = {
    "SOFT":   10,
    "MEDIUM": 20,
    "HARD":   30,
}

T_REF = 30.0

DEG_SCALE = {
    "SOFT":   0.01605558,
    "MEDIUM": 0.00812958,
    "HARD":   0.00404996,
}

DEG_EXPONENT = 1.05463711

# ==============================================================================
# PHYSICS ENGINE
# ==============================================================================

def calculate_lap_time(base_lap_time, compound, tire_age, track_temp):
    laps_past_grace = max(0.0, tire_age - GRACE_LAPS[compound])
    temp_factor     = track_temp / T_REF
    deg_rate        = DEG_SCALE[compound] * (base_lap_time ** DEG_EXPONENT)
    degradation     = deg_rate * laps_past_grace * temp_factor

    return base_lap_time + COMPOUND_OFFSET[compound] + degradation


def simulate_driver(strategy, race_config):
    base_time  = race_config["base_lap_time"]
    pit_time   = race_config["pit_lane_time"]
    total_laps = race_config["total_laps"]
    track_temp = race_config["track_temp"]

    pit_stops  = {s["lap"]: s["to_tire"] for s in strategy.get("pit_stops", [])}
    compound   = strategy["starting_tire"]
    tire_age   = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        tire_age  += 1
        total_time += calculate_lap_time(base_time, compound, tire_age, track_temp)

        if lap in pit_stops:
            compound   = pit_stops[lap]
            tire_age   = 0
            total_time += pit_time

    return total_time

# ==============================================================================
# SIMULATION PIPELINE
# ==============================================================================

def execute_simulation(race_data):
    race_config = race_data["race_config"]
    strategies  = race_data["strategies"]

    rankings = []
    for strat in strategies.values():
        driver_id = strat["driver_id"]
        race_time = simulate_driver(strat, race_config)
        rankings.append((race_time, driver_id))

    rankings.sort()
    return [driver_id for _, driver_id in rankings]

# ==============================================================================
# I/O
# ==============================================================================

def main():
    try:
        raw = sys.stdin.read()
        if not raw:
            return
        race_data = json.loads(raw)
        print(json.dumps({
            "race_id":             race_data["race_id"],
            "finishing_positions": execute_simulation(race_data)
        }))
    except Exception as e:
        sys.stderr.write(f"Simulation Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        print(json.dumps({
            "race_id":             data["race_id"],
            "finishing_positions": execute_simulation(data)
        }))
    else:
        main()