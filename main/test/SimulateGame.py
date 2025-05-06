
"""
Används för att simulera spelet givet en modell.
"""

from open_spiel.python import games
import pyspiel
import pickle
import numpy as np
from absl import app

def simulate_episode(game, policy, model):
    #observer = game.make_py_observer(iig_obs_type=pyspiel.IIGObservationType(perfect_recall=True))
    state = game.new_initial_state()
    while not state.is_terminal():
        cur_player = state.current_player()
        if cur_player == pyspiel.PlayerId.CHANCE:
            # för chance nodes, använd chance outcomes
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            chosen_action = np.random.choice(actions, p=probs)
        else:
            # få sannolikheter för legal actions från policy
            action_probs = policy.action_probabilities(state, cur_player)
            actions, probs = zip(*action_probs.items())
            if model == "D_CFR":
                probs = np.array(probs)
                probs = probs / probs.sum()  # normalisera sannolikheterna (endast för Deep CFR)
            chosen_action = np.random.choice(actions, p=probs)
        state.apply_action(chosen_action)
        print(state)
    print("Final returns:", state.returns())
    return


def main(_):
    
    # filväg till filen, inkludera namnet och filändelse && filnamnet till pickle model att använda
    graph_name = "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/Graf1.csv"
    model_name = "/Users/davidklasa/Documents/GitHub/Kexet/main/PKL_models/Graf1/CFR_model_Graf1.pkl"
    model = "CFR" # CFR | D_CFR | MCCFR

    # laddar in spelet
    params = {
        "filepath": graph_name,
        "filename": "Graf1"
    }
    game = pyspiel.load_game("python_submarine_helicopter", params)

    # laddar in modell från pickle fil
    with open(model_name, "rb") as file:
        loaded_solver = pickle.load(file)

    # simulerar modellen spel
    for _ in range(10):
        simulate_episode(game, loaded_solver, model)


if __name__ == '__main__':
    app.run(main)