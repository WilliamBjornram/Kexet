
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


def main(_):
  # Filväg till grafen, inkludera namnet och filändelse
  filename = "/content/Kexet/main/grafer/Graf0.csv"
  sampling = "external"
  model_data_file = "MCCFR_model.pkl" # filen där den tränade modelen ska sparas
  training_data_file = "MCCFR_training_data.csv" # filen där träningsdatan ska sparas

  info_general = {}
  ind = filename.rfind("/")
  info_general["graph"] = filename[ind:-4]
  s_time = time.time()

  game = pyspiel.load_game("python_submarine_helicopter", dict(filename = filename))
  if sampling == "external":
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
  else:
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)
  
  run_data = []

  e_time = time.time()
  init_tid = e_time - s_time
  info_general["init_t"] = init_tid

  iter = 0 + init_tid
  i = 0
  conv = 1
  while conv >= 0.1:
    c_time = time.time() 
    print(str(i + 1) + " iterations")
    cfr_solver.iteration()
    iter += time.time() - c_time

    if i % 75 == 0:
      print("Evaluation for iteration: " + str(i+1))
      conv = exploitability.nash_conv(game, cfr_solver.average_policy())
      print(conv)
      row = {
                "iteration": i+1,
                "exploitability": conv,
                "graph": info_general["graph"],
                "init_t": info_general["init_t"],
                "tot_t": iter
            }
      run_data.append(row)
      if i == 0:
         with open(training_data_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
            writer.writeheader()
    if i > 1 and i % 100 == 0:
      with open(training_data_file, "a", newline="") as f:
        # Använd fältnamnen från första raden i run_data
        writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
        writer.writerows(run_data)
      run_data = []
    i += 1

  avg_policy = cfr_solver.average_policy()
  with open(model_data_file, "wb") as f:
    pickle.dump(avg_policy, f)
  
  with open(training_data_file, "a", newline="") as f:
      # Använd fältnamnen från första raden i run_data
      writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
      writer.writerows(run_data)

if __name__ == "__main__":
  app.run(main)
