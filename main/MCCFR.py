"""Modified MCCFR experiment to record and average evaluation data across runs.

This script runs the MCCFR experiment multiple times (5 by default). At every evaluation
checkpoint (every 50 iterations), it records the current iteration, exploitability, and total elapsed time.
After all runs complete, it aggregates the results by iteration, computes averages for exploitability
and time, and writes them to a CSV file.
"""

import time
import csv
import pickle
from absl import app
from absl import logging
import os

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python import games
import pyspiel

def run_experiment(filepath, graph_short_name, iter, main_dir, sampling="external"):
    """
    Run one instance of the MCCFR experiment until exploitability drops below 0.1.
    
    During the run, every 50 iterations, record the iteration number, current exploitability,
    and total elapsed time.
    
    Returns:
        run_data (list of dict): List of evaluation records with keys "iteration", 
                                 "exploitability", and "total_time".
        total_run_time (float): Total time taken for the run.
    """

    start_time = time.time()

    params = {
        "filepath": filepath,
        "filename": graph_short_name
    }
    game = pyspiel.load_game("python_submarine_helicopter", params)
    if sampling == "external":
        mccfr_solver = external_mccfr.ExternalSamplingSolver(
            game, external_mccfr.AverageType.SIMPLE)
    else:
        mccfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)

    init_time = time.time() - start_time
    total_iter_time = init_time
    run_data = []

    i = 0
    conv = 1.0

    while conv >= 0.02:
        iter_start = time.time()
        mccfr_solver.iteration()
        total_iter_time += time.time() - iter_start
        logging.info(str(i+1) + " iterations")
        if i % 8 == 0:
            conv = exploitability.nash_conv(game, mccfr_solver.average_policy())
            logging.info(f"Run progress - Iteration {i}, Exploitability: {conv}, Total Time: {total_iter_time:.2f}")
            row = {
                "iteration": i,
                "exploitability": conv,
                "init_tid": init_time,
                "total_time": total_iter_time
            }
            run_data.append(row)
        i += 1

    total_run_time = time.time() - start_time
    logging.info(f"Finished run: Total iterations {i}, Final Exploitability: {conv}, Total Run Time: {total_run_time:.2f} seconds")

    # saving policy with pickle
    pkl_file = os.path.join(main_dir, "PKL_models", graph_short_name, f"MCCFR_model_{graph_short_name}_{iter}")

    avg_policy = mccfr_solver.average_policy()
    with open(pkl_file, "wb") as f:
        pickle.dump(avg_policy, f)
    
    return run_data

def main(_):
    
    graph_short_name = "Graf0"
    main_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(main_dir, "grafer", f"{graph_short_name}.csv")

    sampling = "external"
    num_runs = 5
    all_run_data = []  # List to store evaluation data for each run

    # Run the experiment multiple times.
    for run in range(num_runs):
        logging.info(f"\n=== Starting run {run + 1} ===")
        run_data = run_experiment(filename, graph_short_name, run, main_dir, sampling)

        inter_csv_filenpath = os.path.join(main_dir, "CSV", graph_short_name, f"MCCFR_intermediate_results_{graph_short_name}_{run}.csv")
        with open(inter_csv_filenpath, "w", newline="") as f:
            fieldnames = ["iteration", "graph", "exploitability", "init_tid", "total_time"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(run_data)

        all_run_data.append(run_data)
    
    # Aggregate evaluation data by iteration.
    # We'll assume that all runs record data at the same iteration checkpoints.
    aggregated = {}
    for run_data in all_run_data:
        for row in run_data:
            iter_val = row["iteration"]
            if iter_val not in aggregated:
                aggregated[iter_val] = {"exploitability": [], "init_tid": [], "total_time": []}
            aggregated[iter_val]["exploitability"].append(row["exploitability"])
            aggregated[iter_val]["init_tid"].append(row["init_tid"])
            aggregated[iter_val]["total_time"].append(row["total_time"])

    # Compute averages for each evaluation checkpoint.
    avg_results = []
    for iter_val in sorted(aggregated.keys()):
        avg_exploit = sum(aggregated[iter_val]["exploitability"]) / len(aggregated[iter_val]["exploitability"])
        avg_init_time = sum(aggregated[iter_val]["init_tid"]) / len(aggregated[iter_val]["init_tid"])
        avg_time = sum(aggregated[iter_val]["total_time"]) / len(aggregated[iter_val]["total_time"])
        avg_results.append({
            "iteration": iter_val,
            "average_exploitability": avg_exploit,
            "graph": graph_short_name,
            "average_init_time": avg_init_time,
            "average_total_time": avg_time
        })

    # Write the averaged results to a CSV file.
    csv_filename = os.path.join(main_dir, "CSV", graph_short_name, f"MCCFR_average_results_{graph_short_name}.csv")
    with open(csv_filename, "w", newline="") as f:
        fieldnames = ["iteration", "graph", "average_exploitability", "average_init_time", "average_total_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(avg_results)
    
    logging.info(f"\n=== Average evaluation data written to {csv_filename} ===")

if __name__ == "__main__":
    app.run(main)
