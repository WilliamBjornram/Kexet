"""
Den här filen kör CFR ett visst antal iterationer,
evaluerar i ett bestämt intervall,
sparar average policy mha pickle,
sparar information om körningen och skriver sen till en csv fil.
Information om körningen:
    - tid att initialisera spelträdet
    - tid för varje intervall av iterationer
    - exploitability
    - iteration
    - namn på grafen
"""

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel
import pickle
import time
import csv
from absl import app
from absl import logging
import os

def main(_):

    graph_short_name = "Graf3" # name of graph: Graf2 | Graf3
    main_dir = os.path.dirname(os.path.abspath(__file__)) # dir of this file
    filename = os.path.join(main_dir, "grafer", graph_short_name)
    model_data_file = os.path.join(main_dir, "PKL_models", f"CFR_model_{graph_short_name}.pkl") # file for saving the trained model
    training_data_file = os.path.join(main_dir, "CSV", f"CFR_average_results_{graph_short_name}.csv") # file for saving the data from training

    s_time = time.time() # start time

    # loading game and initializing CFR solver
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
    cfr_solver = cfr.CFRSolver(game)

    e_time = time.time()  # time after intialization
    init_time = e_time - s_time # time for initialization

    iter_time = init_time
    i = 0
    expl = 1
    eval_interv = 3 # interval for evalutating the model

    run_data = [] # list for tracking evaluation data

    while expl >= 0.02:
        logging.info(str(i+1) + " iterations")
        c_time = time.time()  # time before iteration
        cfr_solver.evaluate_and_update_policy()
        iter_time += time.time() - c_time # updating total time of iterations
        # checking if time to evaluate
        if i % eval_interv == 0:
            logging.info("Evaluation for iteration: " + str(i+1))
            logging.info("Total iteration time: " + str(iter_time))
            expl = exploitability.nash_conv(game, cfr_solver.average_policy()) # calculating exploitability
            logging.info("Exploitability: " + str(expl))
            
            # collecting data for evaluation
            row = {
                "iteration": i+1,
                "average_exploitability": expl,
                "graph": graph_short_name,
                "init_time": init_time,
                "average_total_time": iter_time
            }
            run_data.append(row)
            
            # write evaluation data to csv file
            if i == 0:
                with open(training_data_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
                    writer.writeheader()
            
            with open(training_data_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
                writer.writerows(run_data)
            run_data = []
        i += 1

    # saving average policy as a pickle file
    avg_policy = cfr_solver.average_policy()
    with open(model_data_file, "wb") as f:
        pickle.dump(avg_policy, f)

if __name__ == "__main__":
    app.run(main)
