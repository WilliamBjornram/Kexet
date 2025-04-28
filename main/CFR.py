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

    graph_short_name = "L_Graf3.csv"
    main_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(main_dir, "grafer", graph_short_name)
    # Filväg till grafen, inkludera namnet och filändelse
    model_data_file = os.path.join(main_dir, "PKL_models", f"CFR_model_{graph_short_name}") # filen där den tränade modelen ska sparas
    training_data_file = os.path.join(main_dir, "CSV", f"CFR_average_results_{graph_short_name}.csv") # filen där träningsdatan ska sparas

    # starttid för initialisering av spelträdet
    s_time = time.time()

    # Ladda spelet och initialisera CFR
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
    cfr_solver = cfr.CFRSolver(game)

    # Kör CFR iterationer
    e_time = time.time()  # tid efter initialisering
    init_tid = e_time - s_time # tid för initialisering

    iter_time = init_tid
    i = 0
    expl = 1
    eval_interv = 3

    # Lista för att spara data från varje evalueringssteg
    run_data = []

    while expl >= 0.02:
        c_time = time.time()  # tid före iterationerna
        logging.info(str(i+1) + " iterations")
        cfr_solver.evaluate_and_update_policy()
        iter_time += time.time() - c_time
        # När det är dags att utvärdera
        if i % eval_interv == 0:
            logging.info("Evaluation for iteration: " + str(i+1))
            logging.info("Total iteration time: " + str(iter_time))
            expl = exploitability.nash_conv(game, cfr_solver.average_policy())
            logging.info("Exploitability: " + str(expl))
            
            # Samla data för denna evalueringsperiod
            row = {
                "iteration": i+1,
                "average_exploitability": expl,
                "graph": graph_short_name,
                "init_t": init_tid,
                "average_total_time": iter_time
            }
            run_data.append(row)
        
            if i == 0:
                with open(training_data_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
                    writer.writeheader() # när vi skapar filen skriver vi header names
            
            with open(training_data_file, "a", newline="") as f:
                # Använd fältnamnen från första raden i run_data
                writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
                writer.writerows(run_data) # skriv till rad
            run_data = []
        i += 1

    # Spara average policy med pickle
    avg_policy = cfr_solver.average_policy()
    with open(model_data_file, "wb") as f:
        pickle.dump(avg_policy, f)

if __name__ == "__main__":
    app.run(main)
