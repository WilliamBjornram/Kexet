
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
  # Filväg till grafen, inkludera namnet och filändelse
  filename = "/content/Kexet/main/grafer/Test0.csv"
  num_iter = 10   # antal iterationer
  num_traversals = 5  # hur många traversals per iteration
  model_data_file = "Deep_CFR_model.pkl" # filen där den tränade modelen ska sparas
  training_data_file = "Deep_CFR_training_data.csv" # filen där träningsdatan ska sparas

  start_time = time.time()

  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
  
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(16,),
        advantage_network_layers=(16,),
        num_iterations=num_iter,
        num_traversals=num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=1024,
        memory_capacity=1e7,
        policy_network_train_steps=400,
        advantage_network_train_steps=20,
        reinitialize_advantage_networks=False)
    sess.run(tf.global_variables_initializer())
    
    _, advantage_losses, policy_loss = deep_cfr_solver.solve()
    total_run_time = time.time() - start_time

    average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)

    conv = exploitability.nash_conv(game, average_policy)

    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    print("Computed player 0 value: {}".format(average_policy_values[0]))
    print("Computed player 1 value: {}".format(average_policy_values[1]))

  # Save the average policy using pickle.
  with open(model_data_file, "wb") as f:
    pickle.dump(average_policy, f)

  # Save training information (NashConv and total run time) to a CSV file.
  with open(training_data_file, "w", newline="") as csvfile:
    fieldnames = ["nash_conv", "total_run_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"nash_conv": conv, "total_run_time": total_run_time})


if __name__ == "__main__":
  app.run(main)
