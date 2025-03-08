import numpy as np
from absl import app
from absl import flags

from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import exploitability
from open_spiel.python import games
import pyspiel


"""
Den här filen kör CFR ett visst antal gånger,
skriver ut eval ett visst antal ggr,
sen kör den spelet ett visst antal gånger
så man ser hur bra policy man har.
"""

def simulate_episode(game, policy):
    state = game.new_initial_state()
    while not state.is_terminal():
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

# Example usage after CFR training:
game = pyspiel.load_game("python_submarine_helicopter")
cfr_solver = cfr.CFRSolver(game)
# Run CFR iterations...
eval = 25
for i in range(50):
  print("HEJ")
  cfr_solver.evaluate_and_update_policy()
  if i % eval == 0:
    conv = exploitability.exploitability(game, cfr_solver.average_policy())
    print("Iteration {} exploitability {}".format(i, conv))

# Get the average policy and simulate a game.
avg_policy = cfr_solver.average_policy()
for i in range(10):
  simulate_episode(game, avg_policy)