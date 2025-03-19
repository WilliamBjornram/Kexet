
from open_spiel.python import games
import pyspiel
import time

if __name__ == "__main__":
    
    game = pyspiel.load_game("python_submarine_helicopter")

    start = time.time()
    states = pyspiel.get_all_states(game, depth_limit=100, include_terminals=True, include_chance_states=True)
    end = time.time()
    print(f"Total tid tränat: {(end-start)/60} min")
    print("Number of states in game tree:", len(states))