import optuna
import csv
import tensorflow.compat.v1 as tf
from absl import logging
from absl import app
import os
import time

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

# Temporarily disable TF2 behavior.
tf.disable_v2_behavior()

# help function for parsing
def parse_network(network_str):
    # Converts a string like "64,64,64" into a tuple of ints: (64, 64, 64), optuna doesn't handle tuples
    return tuple(int(x.strip()) for x in network_str.split(','))

# help function for 
def tune(traversals, network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps, filepath, trial):

    chunk_iter = 2
    total_chunks = 20
    total_iter = 0

    # Load the game and initialize the solver
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filepath))
    
    with tf.Session() as sess:
        deep_cfr_solver = deep_cfr.DeepCFRSolver(
            sess,
            game,
            policy_network_layers=network,
            advantage_network_layers=network,
            num_iterations=0,
            num_traversals=traversals,
            learning_rate=l_rate,
            batch_size_advantage=b_size_a,
            batch_size_strategy=b_size_p,
            memory_capacity=mem_cap,
            policy_network_train_steps=pn_train_steps,
            advantage_network_train_steps=an_train_steps,
            reinitialize_advantage_networks=True)
        sess.run(tf.global_variables_initializer())

        # iterate and solve
        for _ in range(total_chunks):
            # track time of iteration and update total number of iterations
            chunk_start = time.time()
            total_iter += chunk_iter
            # increase number of iterations and solve
            deep_cfr_solver._num_iterations += chunk_iter 
            _, _, _ = deep_cfr_solver.solve()
            # calculate intermediate exploitability.
            average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
            conv = exploitability.nash_conv(game, average_policy)
            
            # report intermediate result to optuna and if not satisfactory optuna prunes it
            trial.report(conv, total_iter)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            chunk_time = time.time() - chunk_start
            logging.info(f"Iteration {total_iter}: Exploitability = {conv:.10f}, Chunk time = {chunk_time:.2f} sec")
    
    # return nash_conv
    return conv

def objective(trial):
    # Define the search space:
    # Choose number of traversals
    traversals = trial.suggest_categorical("traversals", [10, 20, 50, 100, 200, 300, 400, 500])
    # Choose network architecture from candidate strings.
    network_str = trial.suggest_categorical("network", 
                    ["32,32,32","32,32,32,32", "64,64", "64,64,64", "64,64,64,64", "128,128"])
    network = parse_network(network_str)
    
    # Learning rate: use suggest_float with log scale.
    l_rate = trial.suggest_float("l_rate", 1e-5, 1e-3, log=True)
    # Batch sizes for advantage and strategy:
    b_size_a = trial.suggest_categorical("b_size_a", [256, 512, 1024, 2048])
    b_size_p = trial.suggest_categorical("b_size_p", [256, 512, 1024, 2048])
    # Memory capacity:
    mem_cap = trial.suggest_categorical("mem_cap", [1e6, 2e6, 3e6, 4e6, 1e7])
    # Policy network training steps:
    pn_train_steps = trial.suggest_categorical("pn_train_steps", [256, 1024, 2048, 4096, 8192])
    # Advantage network training steps:
    an_train_steps = trial.suggest_categorical("an_train_steps", [256, 1024, 2048, 4096, 8192])

    # Log the parameters for later analysis
    params = [traversals, network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps]
    logging.info(f"Trial parameters: {params}")

    # getting current dir and name for logging csv file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.dirname(current_dir)
    csv_file = os.path.join(main_dir, "CSV", "DCFR_optuna_Graf2.csv")
    
    # creating graph directory
    filepath = os.path.join(main_dir, "grafer", "Graf2.csv")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file at {filepath} does not exist.")

    # run the tuning procedure
    nash_conv = tune(traversals, network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps, filepath, trial)
    logging.info(f"Trial finished with exploitability: {nash_conv}")

    # write parameters and results to CSV
    row = {
        "traversals": traversals,
        "network": network_str,
        "l_rate": l_rate,
        "b_size_a": b_size_a,
        "b_size_p": b_size_p,
        "mem_cap": mem_cap,
        "pn_train_steps": pn_train_steps,
        "an_train_steps": an_train_steps,
        "exploitability": nash_conv
    }

    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        writer.writerow({})  # Blank row for readability

    return nash_conv

def main(_):
    # creating optuna study with directions to minimize exploitability
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20) # feeding it the helper function and choosing how many trials
    
    # printing best trial at end (with hyperparams)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Exploitability: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    app.run(main)