"""Python Deep CFR example with CSV logging for NashConv and run time."""

import numpy as np
from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf
import csv
import time
import pickle

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

def main(_):
  filepath = "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/Graf0.csv"
  model_data_file = "Deep_CFR_model.pkl"  # file to save the trained model
  training_data_file = "Deep_CFR_training_data.csv"  # file to save training data
  run_data = []
  ind = filepath.rfind("/")
  filename = filepath[ind+1:-4]

  # Define training parameters
  chunk_iter = 10  # number of iterations per training chunk
  total_iter = 0
  threshold = 0.01  # target exploitability threshold
  conv = float('inf')

  # Load the game once
  logging.info("Loading %s", "submarine_helicopter")
  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filepath))

  # Create a single TensorFlow session and initialize the solver once
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(128, 128),
        advantage_network_layers=(128, 128),
        num_iterations=0,  # start with zero iterations
        num_traversals=500,
        learning_rate=1e-3,
        batch_size_advantage=2048,
        batch_size_strategy=2048,
        memory_capacity=1e6,
        policy_network_train_steps=4096,
        advantage_network_train_steps=768,
        reinitialize_advantage_networks=False)
    
    sess.run(tf.global_variables_initializer())

    # Continuous training loop without reinitializing the solver
    while conv >= threshold:
      start_time = time.time()
      # Increment the total iterations by the chunk size
      total_iter += chunk_iter
      # Update the solver's iteration count to run additional iterations
      deep_cfr_solver._num_iterations += chunk_iter
      
      # Run additional training iterations
      _, advantage_losses, policy_loss = deep_cfr_solver.solve()
      
      chunk_run_time = time.time() - start_time

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
      logging.info(f"Total Iterations: {total_iter}, Exploitability: {conv}, Chunk Time: {chunk_run_time} seconds")
      
      average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
      logging.info("Computed sub value: {}".format(average_policy_values[0]))
      logging.info("Computed helicopter value: {}".format(average_policy_values[1]))

      # Log the data
      row = {
          "iteration": total_iter,
          "exploitability": conv,
          "graph": filename,
          "tot_t": chunk_run_time
      }
      run_data.append(row)
      
      # Append to CSV file (write header if first time, else append)
      if total_iter == chunk_iter:
        with open(training_data_file, "w", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=row.keys())
          writer.writeheader()
          writer.writerows(run_data)
          run_data = []
      else:
        with open(training_data_file, "a", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=row.keys())
          writer.writerows(run_data)
          run_data = []

    # Save the final average policy using pickle.
    with open(model_data_file, "wb") as f:
      pickle.dump(average_policy, f)

if __name__ == "__main__":
  app.run(main)