
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

# disabling TF2 behavior temporarily
tf.disable_v2_behavior()

def run_experiment(filename, graph_short_name, iter):

  run_data = [] # to record data during run

  chunk_iter = 10 # number of iterations per training chunk
  total_iter = 1 # to keep track of total iterations
  start_time = time.time()
  tot_run_time = 0

  # load the game
  logging.info("Loading %s", "submarine_helicopter")
  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))  

  # create a TensorFlow session and initialize the solver
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(64, 64, 64),
        advantage_network_layers=(64, 64, 64),
        num_iterations=1,
        num_traversals=int(15e2),
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=4e6,
        policy_network_train_steps=5000,
        advantage_network_train_steps=750,
        reinitialize_advantage_networks=True)
    
    sess.run(tf.global_variables_initializer())

    # loop to train and record results
    for _ in range(11):

      iter_time = time.time()
      
      # run training iterations
      _, advantage_losses, policy_loss = deep_cfr_solver.solve()  
      
      tot_run_time += time.time() - iter_time # how long time did the iterations take

      # logging info for debugging purposes
      for player, losses in advantage_losses.items():
        logging.info("Advantage for player %d: %s", player,
                    losses[:2] + ["..."] + losses[-2:])
        logging.info("Advantage Buffer Size for player %s: '%s'", player,
                    len(deep_cfr_solver.advantage_buffers[player]))
      
      logging.info("Strategy Buffer Size: '%s'",
                len(deep_cfr_solver.strategy_buffer))
      logging.info("Policy loss: '%s'", policy_loss)

      # logging expected game scores
      average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
      average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
      logging.info("Computed game score player 0: {}".format(average_policy_values[0]))
      logging.info("Computed game score player 1: {}".format(average_policy_values[1]))

      # compute average policy from the current solver and exploitability
      conv = exploitability.nash_conv(game, average_policy)
      logging.info(f"Total Iterations: {total_iter}, Exploitability: {conv}, Total Run Time: {tot_run_time} seconds")

      # log the data
      row = {
        "iteration": total_iter,
        "exploitability": conv,
        "graph": graph_short_name,
        "total_time": tot_run_time
        }
      run_data.append(row)
      solo_data = []
      solo_data.append(row)
      
      # print data to intermediate csv file for debugging purposes
      training_data_file = f"DeepCFR_intermediate_results_{iter}.csv"
      if total_iter == 1:
        with open(training_data_file, "w", newline="") as f:
          writer = csv.DictWriter(f, fieldnames=solo_data[0].keys())
          writer.writeheader()

      with open(training_data_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=solo_data[0].keys())
        writer.writerows(solo_data)

      if total_iter == 1:
        deep_cfr_solver._num_iterations = chunk_iter # first do one iteration for benchmark purposes, then chunks
      total_iter += chunk_iter # update total iteration

    # Spara average policy med pickle
    main_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_file = os.path.join(main_dir, "PKL_models", f"DeepCFR_model_{graph_short_name}_{iter}")
  
    with open(pkl_file, "wb") as f:
        pickle.dump(average_policy, f)
      
    return run_data

def main(_):
    # finds graph and specifies on which graph to run game
    graph_short_name = "L_Graf3.csv"
    main_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(main_dir, "grafer", graph_short_name)
    num_runs = 1 # number of runs
    all_run_data = []  # list to store evaluation data for each run

    # run the experiment multiple times
    for run in range(num_runs):
        print(f"\n=== Starting run {run + 1} ===")
        run_data = run_experiment(filename, graph_short_name, run)
        all_run_data.append(run_data)
    
    # aggregates evaluation data by iteration
    # assumes that all runs record data at the same iteration checkpoints
    aggregated = {}
    for run_data in all_run_data:
      for row in run_data:
        iter_val = row["iteration"]
        if iter_val not in aggregated:
            aggregated[iter_val] = {"exploitability": [], "total_time": []}
        aggregated[iter_val]["exploitability"].append(row["exploitability"])
        aggregated[iter_val]["total_time"].append(row["total_time"])

    # computes averages for each evaluation checkpoint
    avg_results = []
    for iter_val in sorted(aggregated.keys()):
        avg_exploit = sum(aggregated[iter_val]["exploitability"]) / len(aggregated[iter_val]["exploitability"])
        avg_time = sum(aggregated[iter_val]["total_time"]) / len(aggregated[iter_val]["total_time"])
        avg_results.append({
            "iteration": iter_val,
            "average_exploitability": avg_exploit,
            "average_total_time": avg_time
        })

    # writes the averaged results to a CSV file
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
