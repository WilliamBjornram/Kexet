
import tensorflow.compat.v1 as tf
import csv
import time
import os
import argparse
from absl import app
from absl import logging
import pickle

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()


def results(filepath, network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps):
  run_data = []
  chunk_iter = 10
  total_iter = 0

  # Loading the game
  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filepath))

  # Starting tensor flow session and initializing DCFR solver
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=network,
        advantage_network_layers=network,
        num_iterations=0,
        num_traversals=500,
        learning_rate=l_rate,
        batch_size_advantage=b_size_a,
        batch_size_strategy=b_size_p,
        memory_capacity=mem_cap,
        policy_network_train_steps=pn_train_steps,
        advantage_network_train_steps=an_train_steps,
        reinitialize_advantage_networks=False)
    
    sess.run(tf.global_variables_initializer())

    for _ in range(6):
      start_time = time.time()
      
      # changing number of iterations, then solving
      deep_cfr_solver._num_iterations += chunk_iter 
      _, _, policy_loss = deep_cfr_solver.solve()
      
      # time for the iterations
      chunk_run_time = time.time() - start_time

      # calculating exploitability
      average_policy = policy.tabular_policy_from_callable(
          game, deep_cfr_solver.action_probabilities)
      conv = exploitability.nash_conv(game, average_policy)
      
      # calculating average values
      average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)

      # save data for the iteration
      row = {
          "iteration": total_iter,
          "exploitability": conv,
          "it_t": chunk_run_time,
          "pl0_buff": len(deep_cfr_solver.advantage_buffers[0]),
          "pl1_buff": len(deep_cfr_solver.advantage_buffers[1]),
          "str_buff": len(deep_cfr_solver.strategy_buffer),
          "policy_loss": policy_loss,
          "avg_game_score_s": average_policy_values[0],
          "avg_game_score_h": average_policy_values[1]
      }
      run_data.append(row)

  return run_data, average_policy


def help_func(filepath, graph_num):
  
  # Hyperparameters for the solver, first index of the list is for 'Graf0', then 'Graf1' and so on
  hyperparams = {
    "network": [(128, 128, 128), (128, 128, 128), (128, 128, 128)],
    "l_rate": [2.4064031872979925e-05, 2.4064031872979925e-05, 2.4064031872979925e-05],
    "b_size_a": [256, 256, 256],
    "b_size_p": [2048, 2048, 2048],
    "mem_cap": [1e7, 1e7, 1e7],
    "pn_train_steps": [8192, 8192, 8192],
    "an_train_steps": [4096, 4096, 4096] 
  }

  # Call results and give them to main
  return results(network=hyperparams["network"][graph_num],
              l_rate=hyperparams["l_rate"][graph_num],
              b_size_a=hyperparams["b_size_a"][graph_num],
              b_size_p=hyperparams["b_size_p"][graph_num],
              mem_cap=hyperparams["mem_cap"][graph_num],
              pn_train_steps=hyperparams["pn_train_steps"][graph_num],
              an_train_steps=hyperparams["an_train_steps"][graph_num],
              filepath=filepath)
  

def main(_):

  # Set up the argument parser
  parser = argparse.ArgumentParser(description="Filepath input")
  parser.add_argument(
      '--filepath',
      type=str,
      default="/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/Graf0.csv",
      help="Full path to the CSV file containing the game Graph"
  )

  # Parse the arguments
  args = parser.parse_args()

  # Use the provided or default filepath
  filepath = args.filepath

  ind = filepath.rfind("/")
  filename = filepath[ind+1:-4]
  graph_num = int(filename[-1])
  csv_filename = "DCFR_data_" + filename + ".csv"

  current_dir = os.path.dirname(os.path.abspath(__file__))
  training_data_file = os.path.join(current_dir, csv_filename)

  logging.info(f"Using Graph: {filename}")

  for i in range(5):
    logging.info(f"Solving {i+1} time....")
    data, average_policy = help_func(filepath=filepath, graph_num=graph_num)

    # Save the average policy using pickle
    logging.info("Saving the pickle model....")
    model_data_file = os.path.join(current_dir, f"DCFR_PKL_models/DCFR_model_{filename}_{i}.pkl")
    with open(model_data_file, "wb") as f:
      pickle.dump(average_policy, f)

    logging.info("Writing data to CSV file....")
    # Append to CSV file (write header if first time, else append)
    if i == 0:
      with open(training_data_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        writer.writerow({})
    else:
      with open(training_data_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writerows(data)
        writer.writerow({})

if __name__ == "__main__":
  app.run(main)