# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the MCCFR algorithm on Kuhn Poker."""

import numpy as np
from absl import app
from absl import flags
from numpy import average

from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import external_sampling_mccfr as external_mccfr
from open_spiel.python.algorithms import outcome_sampling_mccfr as outcome_mccfr
from open_spiel.python import games
import pyspiel
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_enum(
    "sampling",
    "outcome",
    ["external", "outcome"],
    "Sampling for the MCCFR solver",
)
flags.DEFINE_integer("iterations", 100, "Number of iterations")
flags.DEFINE_string("game", "python_submarine_helicopter", "Name of the game")
flags.DEFINE_integer("players", 2, "Number of players")
flags.DEFINE_integer("print_freq", 10,
                     "How often to print the exploitability")

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

def main(_):
  game = pyspiel.load_game(FLAGS.game)
  if FLAGS.sampling == "external":
    cfr_solver = external_mccfr.ExternalSamplingSolver(
        game, external_mccfr.AverageType.SIMPLE)
  else:
    cfr_solver = outcome_mccfr.OutcomeSamplingSolver(game)
  for i in range(FLAGS.iterations):
    cfr_solver.iteration()
    if i % FLAGS.print_freq == 0:
      conv = exploitability.nash_conv(game, cfr_solver.average_policy())
      print("Iteration {} exploitability {}".format(i, conv))
  avg_policy = cfr_solver.average_policy()
  with open("MCCFR_model.pkl", "wb") as f:
    pickle.dump(avg_policy, f)
  for _ in range(10):
    simulate_episode(game, avg_policy)


if __name__ == "__main__":
  app.run(main)
