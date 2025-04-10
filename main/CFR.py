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

def main(_):
    # Filväg till grafen, inkludera namnet och filändelse
    filename = "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/Graf3.csv"
    model_data_file = "CFR_model.pkl" # filen där den tränade modelen ska sparas
    training_data_file = "CFR_average_results.csv" # filen där träningsdatan ska sparas
 
    # Håll koll på generell körinformation
    info_general = {}
    # Hämta grafnamnet (exempelvis från sista "/" till -4 position för att klippa bort filändelsen)
    ind = filename.rfind("/")  # rfind returns the last index of "/"
    info_general["graph"] = filename[ind+1:-4]  # grafens namn
    s_time = time.time()  # starttid för initialisering av spelträdet

    # Ladda spelet och initialisera CFR
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
    cfr_solver = cfr.CFRSolver(game)
    
    # Lista för att spara data från varje evalueringssteg
    run_data = []

    # Kör CFR iterationer
    e_time = time.time()  # tid efter initialisering
    init_tid = e_time - s_time
    info_general["init_t"] = init_tid

    iter = 0 + init_tid
    i = 0
    expl = 1

    while expl >= 0.02:
        c_time = time.time()  # tid före iterationerna
        print(str(i+1) + " iterations")
        cfr_solver.evaluate_and_update_policy()
        iter += time.time() - c_time
        # När det är dags att utvärdera
        if i % 3 == 0:
            print("Evaluation for iteration: " + str(i+1) + "and total iteration time: " + str(iter))
            expl = exploitability.nash_conv(game, cfr_solver.average_policy())
            print(expl)
            # Samla data för denna evalueringsperiod
            row = {
                "iteration": i+1,
                "average_exploitability": expl,
                "graph": info_general["graph"],
                "init_t": info_general["init_t"],
                "average_total_time": iter
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

    # Spara average policy med pickle
    avg_policy = cfr_solver.average_policy()
    with open(model_data_file, "wb") as f:
        pickle.dump(avg_policy, f)

    # Skriv alla evalueringsdata till CSV-filen
    
    with open(training_data_file, "a", newline="") as f:
        # Använd fältnamnen från första raden i run_data
        writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
        writer.writerows(run_data)

if __name__ == "__main__":
    app.run(main)
