
"""
Den här filen kör CFR ett visst antal iterationer,
evaluerar i ett bestämt intervall
sparar average policy mha pickle
sparar information om körningen och skriver sen till en csv fil
information om körningen:
    - tid att initialisera spelträdet
    - tid för varje intervall av iterationer
    - exploitablility
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

def main():
    # filväg till filen, inkludera namnet och filändelse
    filename = "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/GrafLiten.csv"

    # till för att hålla koll på data under körning
    information = {}
    ind = filename.find("/", -1, 0) # letar efter sista /
    information["graph"] = filename[ind:-4] # grafens namn from ind tom -4
    s_time = time.time() # tiden när startar beräkning av spelträdet

    # laddar spelet och initialiserar CFR
    game = pyspiel.load_game("python_submarine_helicopter", dict(filename = filename))
    cfr_solver = cfr.CFRSolver(game)

    e_time = time.time() # tiden när avslutat beräkning av spelträdet
    information["init_t"] = e_time - s_time
    
    # Kör CFR iterationer
    num_iter = 10 # x antal ggr
    eval = 5 # evaluera varje 10:e iteration
    c_time = time.time() # kollar tiden innan börjar köra iterationer
    for i in range(num_iter):
        print("One iteration")
        cfr_solver.evaluate_and_update_policy()
        if i % eval == 0:
            i_time = time.time()
            expl = exploitability.exploitability(game, cfr_solver.average_policy())
            information["iteration_time"] = i_time - c_time # kollar hur lång tid de senaste 'eval' iterationerna tog
            information["exploitability"] = expl # sparar exploitability
            information["iteration"] = i # kollar vilken iteration
            c_time = i_time # uppdaterar

    # spara average policy
    avg_policy = cfr_solver.average_policy()
    with open("CFR_model.pkl", "wb") as f:
        pickle.dump(avg_policy, f)

    csv_file = "training_data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=information[0].keys())
        writer.writeheader()
        writer.writerows(information)

if __name__ == "__main__":
    app.run(main)


##### anteckningar ######
# vad för mer information vill vi spara under iterationerna?
# fixa DeepCFR och MCCFR filerna