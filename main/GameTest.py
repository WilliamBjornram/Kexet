import random
from open_spiel.python import games
import pyspiel

game = pyspiel.load_game("python_submarine_helicopter")
state = game.new_initial_state()

while not state.is_terminal():
    print(state)
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
        num = len(legal_actions)-1
        pos = legal_actions[random.randint(1, num)]
    state.apply_action(pos)

print(state)
print(state.returns())