import json
import sys

# ==============================================================================
# RACE SIMULATION CONSTANTS
# ==============================================================================
# Note: These values were empirically derived from historical race data using
# a differential evolution optimization approach to minimize ranking loss.

# Speed differential per compound relative to base lap time (seconds)
COMPOUND_OFFSET = {
    "SOFT":   -1.0,
    "MEDIUM":  0.0,
    "HARD":    0.8,
}

# Degradation penalty per lap after grace period
DEG_RATE = {
    "SOFT":   1.72,
    "MEDIUM": 0.86,
    "HARD":   0.43,
}

# Number of laps a tire operates at optimal performance before degradation begins
GRACE_LAPS = {
    "SOFT":   10,
    "MEDIUM": 20,
    "HARD":   30,
}

# Reference baseline temperature for scaling degradation factors
T_REF = 30.0

# ==============================================================================
# CORE PHYSICS ENGINE
# ==============================================================================

def calculate_lap_time(base_lap_time, compound, tire_age, track_temp):
    """
    Computes the exact time taken to complete a single lap under current conditions.
    Performance is dictated by the track's base pace, the compound's inherent speed,
    and accumulated thermal/physical degradation.
    """
    compound_pace = COMPOUND_OFFSET[compound]

    # Calculate degradation multiplier based on laps driven past the optimal phase
    laps_degraded = max(0, tire_age - GRACE_LAPS[compound])
    temperature_scaling = track_temp / T_REF
    degradation_penalty = DEG_RATE[compound] * laps_degraded * temperature_scaling

    return base_lap_time + compound_pace + degradation_penalty


def process_driver_strategy(strategy, race_config):
    """
    Simulates a driver's entire race based on their pit strategy.
    Returns the total accumulated race time in seconds.
    """
    base_time = race_config["base_lap_time"]
    pit_penalty = race_config["pit_lane_time"]
    total_laps = race_config["total_laps"]
    track_temp = race_config["track_temp"]

    # Pre-map pit stop laps to the tire compound being fitted
    pit_stops = {stop["lap"]: stop["to_tire"] for stop in strategy.get("pit_stops", [])}

    current_compound = strategy["starting_tire"]
    tire_age = 0
    total_race_time = 0.0

    for current_lap in range(1, total_laps + 1):
        # Tires age by 1 lap before the lap time calculation occurs
        tire_age += 1

        lap_time = calculate_lap_time(base_time, current_compound, tire_age, track_temp)
        total_race_time += lap_time

        # If a stop is scheduled at the end of this lap
        if current_lap in pit_stops:
            current_compound = pit_stops[current_lap]
            tire_age = 0  # Re-zero age (will age to 1 at start of next lap)
            total_race_time += pit_penalty

    return total_race_time

# ==============================================================================
# SIMULATION PIPELINE
# ==============================================================================

def execute_race_simulation(race_data):
    """
    Simulates the race for all grid slots and returns the deterministic finishing order.
    """
    race_config = race_data["race_config"]
    strategies = race_data["strategies"]

    driver_times = []

    for _, strategy in strategies.items():
        driver_id = strategy["driver_id"]
        total_time = process_driver_strategy(strategy, race_config)
        driver_times.append((total_time, driver_id))

    # The winner is the driver with the lowest accumulated race time
    driver_times.sort(key=lambda x: x[0])

    return [driver_id for _, driver_id in driver_times]

# ==============================================================================
# SYSTEM I/O
# ==============================================================================

def run_prediction(input_path=None, output_path=None):
    """Handles standard I/O for production test runners."""
    if input_path:
        with open(input_path, "r") as json_file:
            race_data = json.load(json_file)
    else:
        # Piped standard input for automated competition runners
        input_string = sys.stdin.read()
        race_data = json.loads(input_string)

    predicted_ranking = execute_race_simulation(race_data)

    output = {
        "race_id": race_data["race_id"],
        "finishing_positions": predicted_ranking
    }

    if output_path:
        with open(output_path, "w") as json_file:
            json.dump(output, json_file, indent=2)
    else:
        print(json.dumps(output, indent=2))

    return output


def _run_local_validation(input_path, expected_path):
    """Internal helper to validate predictions against a known solution."""
    with open(input_path, "r") as json_file:
        race_data = json.load(json_file)

    predicted = execute_race_simulation(race_data)
    
    with open(expected_path, "r") as json_file:
        correct = json.load(json_file)["finishing_positions"]

    if predicted == correct:
        print(f"[{race_data['race_id']}] PASSED - Perfect match")
    else:
        print(f"[{race_data['race_id']}] FAILED - Prediction diverged from reality")

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse command line execution modes
    if len(args) == 3 and args[0] == "validate":
        _run_local_validation(args[1], args[2])
    elif len(args) == 2:
        run_prediction(args[0], args[1])
    elif len(args) == 1:
        run_prediction(args[0])
    else:
        # Default pipe execution
        run_prediction()
