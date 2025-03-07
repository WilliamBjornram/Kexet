import random
from open_spiel.python import games
import pyspiel

# Load the game
game = pyspiel.load_game("python_submarine_helicopter")

# Create the observer (with perfect recall or whichever IIGObservationType you need).
observer = game.make_py_observer(iig_obs_type=pyspiel.IIGObservationType(perfect_recall=True))

state = game.new_initial_state()

while not state.is_terminal():
    print(state)
    
    # --- Print observations for all players ---
    for p in range(game.num_players()):
        observer.set_from(state, p)
        obs_string = observer.string_from(state, p)
        print(f"Player {p}'s observation: {obs_string}")
        # If you also want to see the numeric tensor:
        # print(f"Player {p}'s observation tensor: {observer.tensor}")

    # Example action selection (existing logic):
    legal_actions = state.legal_actions()
    pl = state.current_player()
    if pl == 0:
        pos = 0
        flag = True
        x = 0
        while flag:
            if legal_actions[x] > state.sub_pos:
                pos = legal_actions[x]
                flag = False
            x += 1
    elif pl == 1:
        num = len(legal_actions) - 1
        pos = legal_actions[random.randint(1, num)]

    state.apply_action(pos)

print(state)
print("Final returns:", state.returns())