
from absl import logging
import tensorflow.compat.v1 as tf
import csv
import time
import random

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

def tune(network=(128,128), l_rate=1e-3, b_size_a=2048, b_size_p=2048, mem_cap=1e6, pn_train_steps=4096, an_train_steps=768):
  filepath = "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/Graf0.csv"
  run_data = []
  params = [network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps]

  chunk_iter = 10
  total_iter = 0

  game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filepath))

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

    for _ in range(1,6,1):
      start_time = time.time()
      
      deep_cfr_solver._num_iterations += chunk_iter 
      _, _, policy_loss = deep_cfr_solver.solve()
      
      chunk_run_time = time.time() - start_time

      average_policy = policy.tabular_policy_from_callable(
          game, deep_cfr_solver.action_probabilities)
      conv = exploitability.nash_conv(game, average_policy)
      
      average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)

      row = {
          "iteration": total_iter,
          "exploitability": conv,
          "it_t": chunk_run_time,
          "pl0_buff": len(deep_cfr_solver.advantage_buffers[0]),
          "pl1_buff": len(deep_cfr_solver.advantage_buffers[1]),
          "str_buff": len(deep_cfr_solver.strategy_buffer),
          "policy_loss": policy_loss,
          "avg_game_score_s": average_policy_values[0],
          "avg_game_score_h": average_policy_values[1],
          "parameters": params
      }
      run_data.append(row)
  
  return run_data

def main():

  training_data_file = "/Users/davidklasa/Documents/GitHub/Kexet/main/Tuning DCFR/DCFR_tune.py/Deep_CFR_tuning_data.csv"
  
  network_list = [(64, 64,), (64, 64, 64), (64, 64, 64, 64), (128, 128), (128, 128, 128)]
  l_rate_list = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
  mem_cap_list = [1e6, 2e6, 3e6, 4e6, 1e7]
  b_size_a_list = [256, 512, 1024, 2048]
  b_size_p_list = [256, 512, 1024, 2048]
  pn_train_steps_list = [1024, 2048, 4096, 8192]
  an_train_steps_list = [1024, 2048, 4096, 8192]

  for i in range(100):
    logging.info(f"Starting tuning run: {i}")

    try:
      data = tune(network=random.choice(network_list),
                  l_rate=random.choice(l_rate_list),
                  b_size_a=random.choice(b_size_a_list),
                  b_size_p=random.choice(b_size_p_list),
                  mem_cap=random.choice(mem_cap_list),
                  pn_train_steps=random.choice(pn_train_steps_list),
                  an_train_steps=random.choice(an_train_steps_list))
    except:
      data = {}
    
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
  main()