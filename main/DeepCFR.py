# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 100, "Number of iterations")
flags.DEFINE_integer("num_traversals", 20, "Number of traversals/games")


def main(_):
  filename = "/content/Kexet/main/grafer/Test0.csv"
  logging.info("Loading %s", "python_submarine_helicopter")
  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
  
  with tf.Session() as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(16,),
        advantage_network_layers=(16,),
        num_iterations=FLAGS.num_iterations,
        num_traversals=FLAGS.num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=1024,
        memory_capacity=1e7,
        policy_network_train_steps=400,
        advantage_network_train_steps=20,
        reinitialize_advantage_networks=False)
    sess.run(tf.global_variables_initializer())
    
    # Measure the total run time for deep CFR iterations.
    start_time = time.time()
    _, advantage_losses, policy_loss = deep_cfr_solver.solve()
    total_run_time = time.time() - start_time

    for player, losses in advantage_losses.items():
      logging.info("Advantage for player %d: %s", player,
                   losses[:2] + ["..."] + losses[-2:])
      logging.info("Advantage Buffer Size for player %s: '%s'", player,
                   len(deep_cfr_solver.advantage_buffers[player]))
    logging.info("Strategy Buffer Size: '%s'",
                 len(deep_cfr_solver.strategy_buffer))
    logging.info("Final policy loss: '%s'", policy_loss)
    logging.info("Total run time for iterations: %s seconds", total_run_time)

    average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)

    conv = exploitability.nash_conv(game, average_policy)
    logging.info("Deep CFR in '%s' - NashConv: %s", "python_submarine_helicopter", conv)

    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    print("Computed player 0 value: {}".format(average_policy_values[0]))
    print("Computed player 1 value: {}".format(average_policy_values[1]))

  # Save the average policy using pickle.
  with open("Deep_CFR_model.pkl", "wb") as f:
    pickle.dump(average_policy, f)

  # Save training information (NashConv and total run time) to a CSV file.
  csv_file = "Deep_CFR_training_data.csv"
  with open(csv_file, "w", newline="") as csvfile:
    fieldnames = ["nash_conv", "total_run_time"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"nash_conv": conv, "total_run_time": total_run_time})


if __name__ == "__main__":
  app.run(main)
