
"""
Används för att simulera spelet givet en modell.
"""

from open_spiel.python import games
import pyspiel
import pickle
import numpy as np
from absl import app
import os

def simulate_episode(game, policy, model):
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
            if model == "DeepCFR":
                probs = np.array(probs)
                probs = probs / probs.sum()  # normalisera sannolikheterna (endast för Deep CFR)
            chosen_action = np.random.choice(actions, p=probs)
        state.apply_action(chosen_action)
        #print(state)
    #print("Final returns:", state.returns())
    return state


def help_func(graph_short_name, model):

    py_dir = os.path.dirname(os.path.abspath(__file__)) # dir of this file
    main_dir = os.path.dirname(py_dir.rstrip('/'))

    graph_name = os.path.join(main_dir, "grafer", f"{graph_short_name}.csv")
    model_name = os.path.join(main_dir, "PKL_models", graph_short_name, f"{model}_model_{graph_short_name}{"_0" if model != "CFR" else ""}.pkl")

    # laddar in spelet
    params = {
        "filepath": graph_name,
        "filename": graph_short_name
    }

    game = pyspiel.load_game("python_submarine_helicopter", params)

    # laddar in modell från pickle fil
    with open(model_name, "rb") as file:
        loaded_solver = pickle.load(file)

    # simulerar modellen spel
    sub_strategy_dict = {}
    heli_strategy_dict = {}
    for _ in range(1000):
        state = simulate_episode(game, loaded_solver, model)
        history = state.history
        sub_list = []
        heli_list = []
        for player, action in history:
            sub_list.append(action) if player == 0 else heli_list.append(action)

        if f"{sub_list}" not in sub_strategy_dict:
            sub_strategy_dict[f"{sub_list}"] = 1
        else:
            sub_strategy_dict[f"{sub_list}"] += 1
        
        if f"{heli_list}" not in heli_strategy_dict:
            heli_strategy_dict[f"{heli_list}"] = 1
        else:
            heli_strategy_dict[f"{heli_list}"] += 1
    
    return sub_strategy_dict, heli_strategy_dict

    
def main(_):
    
    graph_short_name = ["Graf0", "Graf1"] # name of graph: Graf2 | Graf3
    model = ["CFR", "DeepCFR", "MCCFR"]

    with open("simulation_results.txt", "w") as output_file:
        for m in model:
            s_dict_Graf0, h_dict_Graf0 = help_func(graph_short_name[0], m)
            s_dict_Graf1, h_dict_Graf1 = help_func(graph_short_name[1], m)

            output_file.write(f"Model: {m}\n")
            output_file.write(f"Submarine strategies for {graph_short_name[0]}:\n{s_dict_Graf0}\n")
            output_file.write(f"Helicopter strategies for {graph_short_name[0]}:\n{h_dict_Graf0}\n")
            output_file.write(f"Submarine strategies for {graph_short_name[1]}:\n{s_dict_Graf1}\n")
            output_file.write(f"Helicopter strategies for {graph_short_name[1]}:\n{h_dict_Graf1}\n")
            output_file.write("\n" + "-"*50 + "\n\n")



if __name__ == '__main__':
    app.run(main)