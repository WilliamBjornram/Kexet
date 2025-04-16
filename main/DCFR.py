"""Python Deep CFR example with CSV logging for NashConv and run time."""

from absl import app
from absl import logging
import tensorflow.compat.v1 as tf
import csv
import time
import pickle
import os

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

def run_experiment(filename, iter):
  info_general = {}
  ind = filename.rfind("/")
  info_general["graph"] = filename[ind+1:-4]
  run_data = []


  # Define training parameters
  chunk_iter = 10 # number of iterations per training chunk
  total_iter = 0
  threshold = 0.00  # target exploitability threshold
  conv = float('inf')

  # Load the game once
  logging.info("Loading %s", "submarine_helicopter")

  init_tid = time.time()
  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))

  # Create a single TensorFlow session and initialize the solver once
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(64, 64, 64, 64),
        advantage_network_layers=(64, 64, 64, 64),
        num_iterations=0,  # start with zero iterations
        num_traversals=250,
        learning_rate=8.657179006139824e-05,
        batch_size_advantage=1024,
        batch_size_strategy=256,
        memory_capacity=4e6,
        policy_network_train_steps=8192,
        advantage_network_train_steps=1024,
        reinitialize_advantage_networks=False)
    
    sess.run(tf.global_variables_initializer())

    init_tid = time.time() - init_tid
    tot_run_time = 0
    print("Init tid: " + str(init_tid))

    # Continuous training loop without reinitializing the solver
    while conv >= threshold or total_iter < chunk_iter*2 + 1:
      start_time = time.time()
      # Increment the total iterations by the chunk size
      # Update the solver's iteration count to run additional iterations
      
      total_iter += chunk_iter
      deep_cfr_solver._num_iterations += chunk_iter      
      
      # Run additional training iterations
      _, advantage_losses, policy_loss = deep_cfr_solver.solve()
      
      tot_run_time = time.time() - start_time + init_tid

      for player, losses in advantage_losses.items():
        logging.info("Advantage for player %d: %s", player,
                    losses[:2] + ["..."] + losses[-2:])
        logging.info("Advantage Buffer Size for player %s: '%s'", player,
                    len(deep_cfr_solver.advantage_buffers[player]))
      
      logging.info("Strategy Buffer Size: '%s'",
                len(deep_cfr_solver.strategy_buffer))
      logging.info("Policy loss: '%s'", policy_loss)

      # Compute the average policy from the current solver
      average_policy = policy.tabular_policy_from_callable(
          game, deep_cfr_solver.action_probabilities)
      conv = exploitability.nash_conv(game, average_policy)
      logging.info(f"Total Iterations: {total_iter}, Exploitability: {conv}, Total Run Time: {tot_run_time} seconds")
      
      average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
      logging.info("Computed sub value: {}".format(average_policy_values[0]))
      logging.info("Computed helicopter value: {}".format(average_policy_values[1]))

      # Log the data
      
      row = {
        "iteration": total_iter,
        "exploitability": conv,
        "graph": filename,
        "total_time": tot_run_time
        }
      run_data.append(row)
      solo_data = []
      solo_data.append(row)
      
      training_data_file = "DeepCFR_results.csv"
      if total_iter == 1:
        with open(training_data_file, "w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=solo_data[0].keys())
          writer.writeheader()

      with open(training_data_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solo_data[0].keys())
        writer.writerows(solo_data)

    # Spara average policy med pickle
    main_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_file = os.path.join(main_dir, "PKL_models", f"DeepCFR_model_{info_general['graph']}_{iter}")
  
    with open(pkl_file, "wb") as f:
        pickle.dump(average_policy, f)
      
    return run_data

def main(_):
    filename = "/content/Kexet/main/grafer/LEFTGGraf2.csv"
    num_runs = 1
    all_run_data = []  # List to store evaluation data for each run

    # Run the experiment multiple times.
    for run in range(num_runs):
        print(f"\n=== Starting run {run + 1} ===")
        run_data = run_experiment(filename, run)
        all_run_data.append(run_data)
    
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
    csv_filename = "DeepCFR_average_results.csv"
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
