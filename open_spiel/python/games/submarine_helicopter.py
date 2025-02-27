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

# Lint as python3
"""Submarine Helicopter game implemented in Python with OpenSpiel.

This version is adapted from a Gymnasium implementation.
The game is zero-sum and features imperfect information.
Player 0 (Sub) moves first along legal (neighboring) nodes while incurring a cost.
Player 1 (Heli) then moves (from the full action space).
Terminal conditions are checked after each move.
"""

import enum
import numpy as np
import pyspiel
import random
import math

# kanske ändra här?
class Action(enum.IntEnum):
  PASS = 0
  BET = 1

# Player 0 == Sub, Player 1 == Helicopter
_NUM_PLAYERS = 2

_GAME_TYPE = pyspiel.GameType(
    short_name="python_submarine_helicopter",
    long_name="Python Submarine Helicopter",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC, # bör vi ha EXPLICIT_STOCHASTIC?
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True, # den här och de under behöver vi se över
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)


class SubmarineHelicopterGame(pyspiel.Game):
  """A Python version of the Submarine Helicopter game using OpenSpiel."""

  def __init__(self, params=None, graph=None):
    """Constructor.

    Args:
      params: (optional) dictionary av parametrar
      graph: en instans av graphClass
    """
    self._graph = graph  # laddar in grafen
    self._budget = self.graph.calc_shortest_path() * 2
    max_moves = math.ceil(self._budget/5) # tar budget/5 och rundar uppåt för att få max antal drag
                                    # blir ett worst case scenario där ubåt bara står stilla, till tiden gått ut
    
    # beräknar max antal actions möjliga i en nod
    max_act = 0
    for key in self._graph.adjacency:
      tl = self._graph.adjacency[key]
      if len(tl) > max_act:
        max_act = len(tl)

    _GAME_INFO = pyspiel.GameInfo(
        num_distinct_actions=max_act, # varierar, därför sätter vi till max antal actions
        max_chance_outcomes=2, # när sub.pos == heli.pos chance ger två alternativ
        num_players=_NUM_PLAYERS,
        min_utility=-1.0, # belöning max, min och summa (zero sums)
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=max_moves)

    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a new initial state."""
    return SubmarineHelicopterState(self, self._graph, self._budget)

# förstår inte denna
  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return SubmarineHelicopterObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class SubmarineHelicopterState(pyspiel.State):
  """A Python version of the Submarine Helicopter game state.

  The state stores:
    - sub_pos: the node where the Sub is currently located.
    - heli_pos: the node where the heli is currently located.
    - timer: remaining budget for the Sub.
    - _current_player: whose turn it is (0 for Sub, 1 for Heli).
    - _game_over: flag for terminal state.
  """

  def __init__(self, game, graph, budget):
    """Initializes state.

    The graph is passed from the game. The initial positions and budget are set
    based on the graph.
    """
    super().__init__(game)
    self.graph = graph

    self.timer = budget

    # randomiserar vart ubåt startar
    start_candidates = [node for node, flag in self.graph.start_nodes.items() if flag]
    if not start_candidates:
      raise Exception("No start nodes defined in graph.")
    self.sub_pos = random.choice(start_candidates)

    # startar i någon av noderna [1,2,3] (randomiserat)
    self.heli_pos = random.randint(1, 3)

    self._game_over = False

    # ubåten rör sig först
    self._current_player = 0

    # för att hålla koll på vilka drag som gjorts
    self.history = []

  def current_player(self):
    """returnerar id av den aktuella spelaren annars om spel slut -> terminal"""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    return self._current_player

  def _legal_actions(self, player):
    """returnerar en lista på legala drag för aktuell spelare"""
    if player == 0:
      return self.graph.adjacency[self.sub_pos]
    elif player == 1:
      adj_l = self.graph.adjacency[self.heli_pos]
      return list(self.graph.nodes.keys())
    else:
      return []

  def _apply_action(self, action):
    """genomför action"""

    # håller koll på vilka drag som gjorts
    self.history.append((self._current_player, action))

    if self._current_player == 0:
      # om det är ubåts drag
      key = f"{self.sub_pos}:{action}"
      move_cost = self.graph.weights.get(key, float('inf'))
      self.timer -= move_cost
      if not self.graph.adjacency[self.sub_pos].get(action, False):
        raise Exception(f"Illegalt drag från Ubåt.")
      self.sub_pos = action

      # kolla om spelet är slut
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        # om klart så returnerar vi reward
        self._returns = self._terminal_returns(reward)
        return

      # om inte klart byter tur till nästa spelare
      self._current_player = 1

    elif self._current_player == 1:
      # alla drag giltiga
      self.heli_pos = action

      # samma som ovan
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        self._returns = self._terminal_returns(reward)
      else:
        # byter spelare
        self._current_player = 0

  ##### ta vidare härifrån ########

  def _check_terminal(self):
    """Checks if the state is terminal and returns (terminal_flag, reward).

    Reward is from the Sub's perspective (and game is zero-sum).
      - If Sub reaches an end node, reward = +1.
      - If Sub is caught (Sub and Heli in same node) or timer <= 0, reward = -1.
      - Otherwise, game continues.
    """
    # om ubåt vid slutnod -> spelet slut
    if self.graph.end_nodes.get(self.sub_pos, 0) == 1:
      return True, +1
    # 
    if self.sub_pos == self.heli_pos:
      num = random.randint(1, 10)
      if num > self.graph.discovery[self.sub_pos]:
        return False, 0
      else:
        return True, -1
    # om timer är slut -> negativ belöning
    if self.timer <= 0:
      return True, -1
    return False, 0

  def _terminal_returns(self, sub_reward):
    """returnerar vector för belöning,
    eftersom spelet är zero-sum, så blir Heli's belöning negativ
    """
    return [sub_reward, -sub_reward]

  def is_terminal(self):
    """returnerar True om spelet är över"""
    return self._game_over

  def returns(self):
    """returnerar belöning om state är terminal,
    annars om inte terminal -> 0
    """
    if not self.is_terminal():
      return [0.0, 0.0]
    return self._returns

  def __str__(self):
    """Returns a string representation of the state."""
    return (f"Submarine at {self.sub_pos}, Heli at {self.heli_pos}, "
            f"Timer: {self.timer:.1f}, History: {self.history}")


class SubmarineHelicopterObserver:
  """Observer for the Submarine Helicopter game state.

  For simplicity, we build a flat observation vector consisting of:
    - One-hot encoding of the Sub's current node.
    - One-hot encoding of the Heli's current node.
    - A normalized timer value.
  """
  def __init__(self, iig_obs_type, params):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Assume graph has N nodes. We will set N later when observing.
    self.tensor = None
    self.iig_obs_type = iig_obs_type

  def set_from(self, state, player):
    """Update the observation tensor from the state."""
    # Let N be the number of nodes.
    N = len(state.graph.nodes)
    # We construct an observation vector: [sub_one_hot, heli_one_hot, timer]
    obs = np.zeros(2 * N + 1, dtype=np.float32)
    obs[state.sub_pos] = 1.0  # one-hot for Sub's position.
    obs[N + state.heli_pos] = 1.0  # one-hot for Heli's position.
    # Normalize timer by budget.
    obs[-1] = state.timer / state.budget
    self.tensor = obs

  def string_from(self, state, player):
    """Returns a string representation of the observation."""
    return (f"Sub:{state.sub_pos} Heli:{state.heli_pos} Timer:{state.timer:.1f}")


# Register the game with OpenSpiel.
pyspiel.register_game(_GAME_TYPE, SubmarineHelicopterGame)