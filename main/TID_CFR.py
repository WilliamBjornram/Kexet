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
    filename = "/content/Kexet/main/grafer/Test0.csv"
    num_iter = 101   # antal iterationer
    model_data_file = "CFR_model.pkl" # filen där den tränade modelen ska sparas
    training_data_file = "CFR_training_data.csv" # filen där träningsdatan ska sparas

    # Håll koll på generell körinformation
    info_general = {}
    # Hämta grafnamnet (exempelvis från sista "/" till -4 position för att klippa bort filändelsen)
    ind = filename.rfind("/")  # rfind returns the last index of "/"
    info_general["graph"] = filename[ind:-4]  # grafens namn
    s_time = time.time()  # starttid för initialisering av spelträdet

    # Ladda spelet och initialisera CFR
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename=filename))
    cfr_solver = cfr.CFRSolver(game)
    e_time = time.time()  # tid efter initialisering
    info_general["init_t"] = e_time - s_time

    # Lista för att spara data från varje evalueringssteg
    run_data = []

    # Kör CFR iterationer

    for i in range(num_iter):
        print(str(i+1) + " iterations")
        cfr_solver.evaluate_and_update_policy()
    c_time = time.time()

    info_general["tot_time"] = c_time - s_time
    conv = exploitability.nash_conv(game, cfr_solver.average_policy())
    row = {
                "exploitability": conv,
                "graph": info_general["graph"],
                "init_t": info_general["init_t"],
                "tot_t": info_general["tot_time"]
            }
    run_data.append(row)
    # Spara average policy med pickle
    avg_policy = cfr_solver.average_policy()
    with open(model_data_file, "wb") as f:
        pickle.dump(avg_policy, f)

    # Skriv alla evalueringsdata till CSV-filen
    
    with open(training_data_file, "w", newline="") as f:
        # Använd fältnamnen från första raden i run_data
        writer = csv.DictWriter(f, fieldnames=run_data[0].keys())
        writer.writeheader()
        writer.writerows(run_data)

if __name__ == "__main__":
    app.run(main)
