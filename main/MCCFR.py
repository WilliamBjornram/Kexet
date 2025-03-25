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

"""Example use of the MCCFR algorithm on Kuhn Poker."""

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

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "sampling",
    "outcome",
    ["external", "outcome"],
    "Sampling for the MCCFR solver",
)

def main(_):
  # Filväg till grafen, inkludera namnet och filändelse
  filename = "/content/Kexet/main/grafer/Test0.csv"
  num_iter = 101   # antal iterationer
  eval_interval = 5  # hur ofta vi ska evaluera
  model_data_file = "CFR_model.pkl" # filen där den tränade modelen ska sparas
  training_data_file = "CFR_training_data.csv" # filen där träningsdatan ska sparas

  info_general = {}
  ind = filename.rfind("/")
  info_general["graph"] = filename[ind:-4]
  s_time = time.time()
  game = pyspiel.load_game("python_submarine_helicopter", dict(filename = filename))
  if FLAGS.sampling == "external":
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
  else:
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)
  
  e_time = time.time()
  info_general["init_t"] = e_time - s_time

  run_data = []
  c_time = time.time() 
  for i in range(num_iter):
    print("One iteration")
    cfr_solver.iteration()

    if i % eval_interval == 0:
      i_time = time.time()
      conv = exploitability.nash_conv(game, cfr_solver.average_policy())
      row = {
                "iteration": i,
                "iteration_time": i_time - c_time,
                "exploitability": conv,
                "graph": info_general["graph"],
                "init_t": info_general["init_t"]
            }
      run_data.append(row)
      c_time = i_time
  avg_policy = cfr_solver.average_policy()
  with open("MCCFR_model.pkl", "wb") as f:
    pickle.dump(avg_policy, f)
  
  csv_file = "MCCFR_training_data.csv"
  with open(csv_file, "w", newline="") as f:
      # Använd fältnamnen från första raden i run_data
      writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
      writer.writeheader()
      writer.writerows(run_data)

if __name__ == "__main__":
  app.run(main)
