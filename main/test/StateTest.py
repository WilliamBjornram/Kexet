"""
Detta test beräknar hela spelträdet och tar tid på processen.
"""

from open_spiel.python import games
from open_spiel.python.algorithms.get_all_states import get_all_states
import pyspiel
import time

if __name__ == "__main__":
    
    game = pyspiel.load_game("python_submarine_helicopter")

    start = time.time()
    states = get_all_states(game, depth_limit=100, include_terminals=True, include_chance_states=True)
    end = time.time()
    print(f"Total tid: {(end-start)/60} min")
    print("Number of states in game tree:", len(states))