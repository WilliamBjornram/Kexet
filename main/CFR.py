
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel
import pickle
import time
import csv
import numpy as np

"""
Den här filen kör CFR ett visst antal gånger,
skriver ut eval ett visst antal ggr,
sen kör den spelet ett visst antal gånger
så man ser hur bra policy man har.
"""

def simulate_episode(game, policy):
    observer = game.make_py_observer(iig_obs_type=pyspiel.IIGObservationType(perfect_recall=True))
    state = game.new_initial_state()
    while not state.is_terminal():
        """
        for p in range(game.num_players()):
          observer.set_from(state, p)
          obs_string = observer.string_from(state, p)
          print(f"Player {p}'s observation: {obs_string}")
          # If you also want to see the numeric tensor:
          # print(f"Player {p}'s observation tensor: {observer.tensor}")
        """
        cur_player = state.current_player()
        if cur_player == pyspiel.PlayerId.CHANCE:
            # For chance nodes, use the provided chance outcomes.
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            chosen_action = np.random.choice(actions, p=probs)
        else:
            # Get the probabilities for legal actions from the policy.
            action_probs = policy.action_probabilities(state, cur_player)
            actions, probs = zip(*action_probs.items())
            chosen_action = np.random.choice(actions, p=probs)
        state.apply_action(chosen_action)
        print(state)
    print("Final returns:", state.returns())
    return


if __name__ == "__main__":
    # filväg till filen, inkludera namnet och filändelse
    filename = "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/GrafLiten.csv"

    # till för att hålla koll på data under körning
    information = {}
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
        cfr_solver.evaluate_and_update_policy()
        if i % eval == 0:
            i_time = time.time()
            conv = exploitability.exploitability(game, cfr_solver.average_policy())
            information["iteration_time"] = i_time - c_time # kollar hur lång tid de senaste 'eval' iterationerna tog
            information["exploitability"] = conv # sparar exploitability
            information["iteration"] = i # kollar vilken iteration
            c_time = i_time # uppdaterar

    # spara average policy
    avg_policy = cfr_solver.average_policy()
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(avg_policy, f)

    csv_file = "training_data.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=information[0].keys())
        writer.writeheader()
        writer.writerows(information)

    for i in range(10):
        simulate_episode(game, avg_policy)


##### anteckningar ######
# ska vi simulera episoder efter vi har tränat?
# vad för mer information vill vi spara under iterationerna?