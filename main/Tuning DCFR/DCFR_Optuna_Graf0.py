import optuna
import csv
import tensorflow.compat.v1 as tf
from absl import logging
from absl import app
import os

from open_spiel.python import policy
from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import exploitability
import pyspiel

# Temporarily disable TF2 behavior.
tf.disable_v2_behavior()

# Help function
def parse_network(network_str):
    # Converts a string like "64,64,64" into a tuple of ints: (64, 64, 64), optuna doesn't handle tuples
    return tuple(int(x.strip()) for x in network_str.split(','))

def tune(network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps, filepath):
    chunk_iter = 10 # amount of iterations between each evaluation

    # loading the game
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filepath))
    
    # staring a tensor flow session and initializing DCFR solver
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

        # Run for a fixed number of iterations
        for _ in range(6):
            deep_cfr_solver._num_iterations += chunk_iter 
            _, _, _ = deep_cfr_solver.solve()
        
        # Calculate exploitability
        average_policy = policy.tabular_policy_from_callable(game, deep_cfr_solver.action_probabilities)
        conv = exploitability.nash_conv(game, average_policy)
    
    # Return the final exploitability as the objective (lower is better)
    return conv

def objective(trial):
    # Define the search space:
    # Choose network architecture from candidate strings.
    network_str = trial.suggest_categorical("network", 
                    ["64,64", "64,64,64", "64,64,64,64", "128,128", "128,128,128"])
    network = parse_network(network_str)
    
    # Learning rate: use suggest_float with log scale.
    l_rate = trial.suggest_float("l_rate", 1e-5, 1e-3, log=True)
    # Batch sizes for advantage and strategy:
    b_size_a = trial.suggest_categorical("b_size_a", [256, 512, 1024, 2048])
    b_size_p = trial.suggest_categorical("b_size_p", [256, 512, 1024, 2048])
    # Memory capacity:
    mem_cap = trial.suggest_categorical("mem_cap", [1e6, 2e6, 3e6, 4e6, 1e7])
    # Policy network training steps:
    pn_train_steps = trial.suggest_categorical("pn_train_steps", [1024, 2048, 4096, 8192])
    # Advantage network training steps:
    an_train_steps = trial.suggest_categorical("an_train_steps", [1024, 2048, 4096, 8192])

    # Log the parameters for later analysis
    params = [network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps]
    logging.info(f"Trial parameters: {params}")

    # getting current dir and name for logging csv file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "DCFR_optuna_Graf2.csv")
    
    # finding/creating graph directory
    main_dir = os.path.dirname(current_dir)
    filepath = os.path.join(main_dir, "grafer", "Graf2.csv")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"The file at {filepath} does not exist.")

    # Run the tuning procedure
    exploitability_value = tune(network, l_rate, b_size_a, b_size_p, mem_cap, pn_train_steps, an_train_steps, filepath)
    logging.info(f"Trial finished with exploitability: {exploitability_value}")

    # Write parameters and results to CSV
    row = {
        "network": network_str,
        "l_rate": l_rate,
        "b_size_a": b_size_a,
        "b_size_p": b_size_p,
        "mem_cap": mem_cap,
        "pn_train_steps": pn_train_steps,
        "an_train_steps": an_train_steps,
        "exploitability": exploitability_value
    }
    with open(csv_file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        writer.writerow({})  # Blank row for readability

    return exploitability_value

def main(_):
    # Creating optuna study with directions to minimize exploitability
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10) # feeding it the helper function, no more than 10 trials
    
    # printing best trial at end (with hyperparams)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Exploitability: {trial.value}")
    print("  Best hyperparameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    app.run(main)