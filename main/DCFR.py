from absl import app
from absl import logging
import tensorflow.compat.v1 as tf
import csv
import time
import pickle
import os
import multiprocessing

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel

# disabling TF2 behavior temporarily
tf.disable_v2_behavior()

def run_experiment(filepath, graph_short_name, iter, main_dir):

  run_data = [] # to record data during run

  chunk_iter = 10 # number of iterations per training chunk
  total_iter = 1 # to keep track of total iterations
  start_time = time.time()
  tot_run_time = 0.0
  tot_learn_time = 0.0
  conv = 1.0
  pkl_file = os.path.join(main_dir, "PKL_models", graph_short_name, f"DeepCFR_model_{graph_short_name}_{iter}.pkl")

  # load the game
  #logging.info("Loading %s", "submarine_helicopter")
  params = {
    "filepath": filepath,
    "filename": graph_short_name
  }
  game = pyspiel.load_game("python_submarine_helicopter", params)

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

    while conv >= 0.02 and time.time() - start_time < float(86400):

      iter_time = time.time()
      
      # run training iterations
      _, advantage_losses, policy_loss, learn_time = deep_cfr_solver.solve()  
      
      tot_run_time += time.time() - iter_time # how long time did the iterations take
      tot_learn_time += learn_time

      """
      # logging info for debugging purposes
      for player, losses in advantage_losses.items():
        logging.info("Advantage for player %d: %s", player,
                    losses[:2] + ["..."] + losses[-2:])
        logging.info("Advantage Buffer Size for player %s: '%s'", player,
                    len(deep_cfr_solver.advantage_buffers[player]))
      
      logging.info("Strategy Buffer Size: '%s'",
                len(deep_cfr_solver.strategy_buffer))
      logging.info("Policy loss: '%s'", policy_loss)
      """

      # logging expected game scores
      average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)

      """
      average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
      logging.info("Computed game score player 0: {}".format(average_policy_values[0]))
      logging.info("Computed game score player 1: {}".format(average_policy_values[1]))
      """

      # compute average policy from the current solver and exploitability
      conv = exploitability.nash_conv(game, average_policy)
      #logging.info(f"Total Iterations: {total_iter}, Exploitability: {conv}, Total Run Time: {tot_run_time} seconds")

      # log the data
      row = {
        "iteration": total_iter,
        "exploitability": conv,
        "graph": graph_short_name,
        "total_time": tot_run_time,
        "total_learn_time": tot_learn_time
      }
      run_data.append(row)
      solo_data = []
      solo_data.append(row)
      
      # write data to intermediate csv file for debugging purposes
      training_data_file = os.path.join(main_dir, "CSV", graph_short_name, f"DeepCFR_intermediate_results_{graph_short_name}_{iter}.csv")
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

    # saving policy of current iteration with pickle
    with open(pkl_file, "wb") as f:
        pickle.dump(average_policy, f)
      
    return run_data

def main(_):
    # finds graph and specifies on which graph to run game
    graph_short_name = "Graf1"
    main_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(main_dir, "grafer", f"{graph_short_name}.csv")
    num_runs = 5 # number of runs
    all_run_data = []  # list to store evaluation data for each run

    """
    # run the experiment multiple times
    for run in range(num_runs):
        logging.info(f"\n=== Starting run {run + 1} ===")
        run_data = run_experiment(filename, graph_short_name, run, main_dir)
        all_run_data.append(run_data)
    """

    # Prepare arguments for parallel execution
    args = [
        (filename, graph_short_name, run, main_dir)
        for run in range(num_runs)
    ]
    # Execute runs in parallel using multiprocessing Pool
    with multiprocessing.Pool() as pool:
        all_run_data = pool.starmap(run_experiment, args)
    
    # aggregates evaluation data by iteration
    # assumes that all runs record data at the same iteration checkpoints
    aggregated = {}
    for run_data in all_run_data:
      for row in run_data:
        iter_val = row["iteration"]
        if iter_val not in aggregated:
            aggregated[iter_val] = {
              "exploitability": [], 
              "total_time": [], 
              "total_learn_time": [], 
              "graph": []
            }
        aggregated[iter_val]["exploitability"].append(row["exploitability"])
        aggregated[iter_val]["total_time"].append(row["total_time"])
        aggregated[iter_val]["total_learn_time"].append(row["total_learn_time"])
        aggregated[iter_val]["graph"].append(row["graph"])

    # computes averages for each evaluation checkpoint
    avg_results = []
    for iter_val in sorted(aggregated.keys()):
        avg_exploit = sum(aggregated[iter_val]["exploitability"]) / len(aggregated[iter_val]["exploitability"])
        avg_time = sum(aggregated[iter_val]["total_time"]) / len(aggregated[iter_val]["total_time"])
        avg_learn = sum(aggregated[iter_val]["total_learn_time"]) / len(aggregated[iter_val]["total_learn_time"])
        graph_name = aggregated[iter_val]["graph"][0] if aggregated[iter_val]["graph"] else ""
        avg_results.append({
            "iteration": iter_val,
            "exploitability": avg_exploit,
            "total_time": avg_time,
            "total_learn_time": avg_learn,
            "graph": graph_name
        })

    # writes the averaged results to a CSV file
    csv_filename = os.path.join(main_dir, "CSV", graph_short_name, f"DeepCFR_average_results_{graph_short_name}.csv")
    with open(csv_filename, "w", newline="") as f:
        fieldnames = ["iteration", "exploitability", "total_time", "total_learn_time", "graph"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(avg_results)
    
    logging.info(f"\n=== Average evaluation data written to {csv_filename} ===")

if __name__ == "__main__":
    app.run(main)
