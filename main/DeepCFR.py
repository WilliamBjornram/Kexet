
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
  model_data_file = "Deep_CFR_model.pkl" # filen där den tränade modelen ska sparas
  training_data_file = "Deep_CFR_training_data.csv" # filen där träningsdatan ska sparas
  run_data = []
  ind = filepath.rfind("/")
  filename = filepath[ind+1:-4]

  conv = 1
  i = 1
  while conv >= 0.1:
    # Filväg till grafen, inkludera namnet och filändelse
    
    num_iter = 10 * (i) # antal iterationer
    num_traversals = 1000  # hur många traversals per iteration
    
    start_time = time.time()

    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filepath))
    
    with tf.Session() as sess:
      deep_cfr_solver = deep_cfr.DeepCFRSolver(
          sess,
          game,
          policy_network_layers=(64,64,64),
          advantage_network_layers=(64,64,64),
          num_iterations=num_iter,
          num_traversals=num_traversals,
          learning_rate=1e-4,
          batch_size_advantage=1024,
          batch_size_strategy=2048,
          memory_capacity=1e6,
          policy_network_train_steps=512,
          advantage_network_train_steps=256,
          reinitialize_advantage_networks=False)
      sess.run(tf.global_variables_initializer())
      
      _, advantage_losses, policy_loss = deep_cfr_solver.solve()
      Iter_run_time = time.time() - start_time

      average_policy = policy.tabular_policy_from_callable(
          game, deep_cfr_solver.action_probabilities)

      conv = exploitability.nash_conv(game, average_policy)
      print(conv)
      print("Iterations done: " + str(num_iter))
      row = {
                "iteration": num_iter,
                "exploitability": conv,
                "graph": filename,
                "tot_t": Iter_run_time
      }
      
      run_data.append(row)
      if i == 1:
        with open(training_data_file, "w", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=run_data[0].keys())
          writer.writeheader()
          writer.writerows(run_data)
          run_data = []
      else:
        with open(training_data_file, "a", newline="") as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=run_data[0].keys())
          writer.writerows(run_data)
          run_data = []
      i += 1


  # Save the average policy using pickle.
  with open(model_data_file, "wb") as f:
    pickle.dump(average_policy, f)


if __name__ == "__main__":
  app.run(main)
