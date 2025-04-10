"""Modified MCCFR experiment to record and average evaluation data across runs.

This script runs the MCCFR experiment multiple times (5 by default). At every evaluation
checkpoint (every 50 iterations), it records the current iteration, exploitability, and total elapsed time.
After all runs complete, it aggregates the results by iteration, computes averages for exploitability
and time, and writes them to a CSV file.
"""

import numpy as np
import time
import csv
import pickle
from absl import app
from absl import flags
from numpy import average

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python import games
import pyspiel

def run_experiment(filename, sampling="external"):
    """
    Run one instance of the MCCFR experiment until exploitability drops below 0.1.
    
    During the run, every 50 iterations, record the iteration number, current exploitability,
    and total elapsed time.
    
    Returns:
        run_data (list of dict): List of evaluation records with keys "iteration", 
                                 "exploitability", and "total_time".
        total_run_time (float): Total time taken for the run.
    """
    info_general = {}
    ind = filename.rfind("/")
    info_general["graph"] = filename[ind+1:-4]
    start_time = time.time()

    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
    if sampling == "external":
        cfr_solver = external_mccfr.ExternalSamplingSolver(
            game, external_mccfr.AverageType.SIMPLE)
    else:
        cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)

    run_data = []
    init_time = time.time() - start_time
    total_iter_time = init_time

    i = 0
    conv = 1.0  # initial exploitability value (must be >= 0.05 to start)
    
    max_tid = 10

    while conv >= 0.05 and total_iter_time <= max_tid:
        iter_start = time.time()
        cfr_solver.iteration()
        total_iter_time += time.time() - iter_start
        i += 1

        if i % 100 == 0 or total_iter_time >= max_tid:
            conv = exploitability.nash_conv(game, cfr_solver.average_policy())
            print(f"Run progress - Iteration {i}, Exploitability: {conv}, Total Time: {total_iter_time:.2f}")
            row = {
                "iteration": i,
                "exploitability": conv,
                "total_time": total_iter_time
            }
            run_data.append(row)

    total_run_time = time.time() - start_time
    print(f"Finished run: Total iterations {i}, Final Exploitability: {conv}, Total Run Time: {total_run_time:.2f} seconds")
    
    return run_data, total_run_time

def main(_):
    filename = "/content/Kexet/main/grafer/Graf0.csv"
    sampling = "external"
    num_runs = 5
    all_run_data = []  # List to store evaluation data for each run

    # Run the experiment multiple times.
    for run in range(num_runs):
        print(f"\n=== Starting run {run + 1} ===")
        run_data, run_time = run_experiment(filename, sampling)
        all_run_data.append(run_data)
        print(f"Run {run + 1} complete: Run Time = {run_time:.2f} seconds")
    
    # Aggregate evaluation data by iteration.
    # We'll assume that all runs record data at the same iteration checkpoints.
    aggregated = {}
    for run_data in all_run_data:
        for row in run_data:
            iter_val = row["iteration"]
            if iter_val not in aggregated:
                aggregated[iter_val] = {"exploitability": [], "total_time": []}
            aggregated[iter_val]["exploitability"].append(row["exploitability"])
            aggregated[iter_val]["total_time"].append(row["total_time"])

    # Compute averages for each evaluation checkpoint.
    avg_results = []
    for iter_val in sorted(aggregated.keys()):
        avg_exploit = sum(aggregated[iter_val]["exploitability"]) / len(aggregated[iter_val]["exploitability"])
        avg_time = sum(aggregated[iter_val]["total_time"]) / len(aggregated[iter_val]["total_time"])
        avg_results.append({
            "iteration": iter_val,
            "average_exploitability": avg_exploit,
            "average_total_time": avg_time
        })

    # Write the averaged results to a CSV file.
    csv_filename = "MCCFR_average_results.csv"
    with open(csv_filename, "w", newline="") as f:
        fieldnames = ["iteration", "average_exploitability", "average_total_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(avg_results)
    
    print(f"\n=== Average evaluation data written to {csv_filename} ===")
    for row in avg_results:
        print(row)

if __name__ == "__main__":
    app.run(main)
