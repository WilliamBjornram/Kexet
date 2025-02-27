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

The game is zero-sum and features imperfect information.
Player 0 (Sub) moves first along legal (neighboring) nodes while incurring a cost.
Player 1 (Heli) then moves 1 or 2 nodes.
Terminal conditions are checked after each move.
"""


import numpy as np
import pyspiel
import random
import math

# Player 0 == Sub, Player 1 == Helicopter
_NUM_PLAYERS = 2

_GAME_TYPE = pyspiel.GameType(
    short_name="python_submarine_helicopter",
    long_name="Python Submarine Helicopter",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False)


class SubmarineHelicopterGame(pyspiel.Game):
  """en Python version av spelet Submarine Helicopter mha OpenSpiel."""

  def __init__(self, params=None, graph=None):
    """konstruktor

    Args:
      params: (optional) dictionary av parametrar
      graph: en instans av graphClass
    """
    self._graph = graph  # laddar in grafen
    self._budget = self.graph.calc_shortest_path() * 2
    max_moves = math.ceil(self._budget/5) # tar budget/5 och rundar uppåt för att få max antal drag
                                    # blir ett worst case scenario där ubåt bara står stilla, till tiden gått ut
    
    # sätter action space till all noder, vi ger sen det subset som är aktuellt i varje nod
    max_act = len(self._graph.nodes)

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
    """returnerar ett objekt med återställt state"""
    return SubmarineHelicopterState(self, self._graph, self._budget)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """returnerar ett objekt för observation"""
    return SubmarineHelicopterObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class SubmarineHelicopterState(pyspiel.State):
  """
  State håller koll på:
    - sub_pos: node_id för ubåtens nuvarande position
    - heli_pos: node_id för helikopterns nuvarande position
    - timer: återstående budget för ubåten
    - _current_player: vems tur det är
    - _game_over: flagga för om spelet är över
  """

  def __init__(self, game, graph, budget):
    """initialiserar spelet"""
    super().__init__(game)
    self.graph = graph

    self.buget = budget
    self.timer = budget

    # randomiserar vart ubåt startar
    if not self.graph.start_nodes:
      raise Exception("Inga startnoder definerade i grafen.")
    self.sub_pos = random.choice(self.graph.start_nodes)

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
      return self.heli_act() # hjälp metod för att returnera lista med drag för heli
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
      if action not in self._legal_actions(self._current_player):
        raise Exception("Illegalt drag från Ubåt.")
      self.sub_pos = action

      # kolla om spelet är slut
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        # om klart så returnerar vi reward
        self._returns = [reward, -reward]
        return

      # byter tur till nästa spelare
      self._current_player = 1
      
    elif self._current_player == 1:
      # kollar så gör legalt drag
      if action not in self._legal_actions(self._current_player):
        raise Exception("Illegalt drag från Helikopter.")
      self.heli_pos = action

      # samma som ovan
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        self._returns = [reward, -reward]
      
      # byter spelare
      self._current_player = 0

  def _check_terminal(self):
    """kollar om episoden är slut och returnerar (terminal_flag, belöning).
    belöning är från ubåtens perspektiv (och spelet är zero-sum, så omvända belöningen är helikopterns).
    """
    # om ubåt.pos == heli.pos spelet kan ta slut
    if self.sub_pos == self.heli_pos:
      num = random.randint(1, 10)
      if num > self.graph.discovery[self.sub_pos]:
        return False, 0
      else:
        return True, -1
    # om ubåt vid slutnod -> spelet slut
    if self.sub_pos in self.graph.end_nodes:
      return True, +1
    # om timer är slut -> negativ belöning
    if self.timer <= 0:
      return True, -1
    return False, 0

  def returns(self):
    """returnerar belöning om state är terminal,
    annars om inte terminal -> 0
    """
    if not self._game_over:
      return [0.0, 0.0]
    return self._returns
  
  def is_terminal(self):
    """returnerar True om spelet är över (obligatoriskt)"""
    return self._game_over

  def __str__(self):
    """returnerar en string som representation över state"""
    return (f"Sub: {self.sub_pos}, Heli: {self.heli_pos}, "
            f"Timer: {self.timer:.1f}, History: {self.history}")
  
  def heli_act(self):
    """blir en unik lista med alla neighbors och indirekta neighbors (två steg)"""
    output = set()
    adj_l = self.graph.adjacency[self.heli_pos]
    for entry in adj_l:
      tl = self.graph.adjacency[entry]
      output.add(entry)
      for x in tl:
        output.add(x)
    return list(output)


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
    N = len(state.graph.nodes)
    # Create an observation vector of length 2*N + 1.
    obs = np.zeros(2 * N + 1, dtype=np.float32)
    if player == 0:
      # Player 0 sees its own (Sub's) position.
      obs[state.sub_pos] = 1.0  
      # Optionally, fill the opponent's part with zeros or a default value.
    elif player == 1:
      # Player 1 sees its own (Heli's) position.
      obs[N + state.heli_pos] = 1.0
    # Public info: the timer.
    obs[-1] = state.timer/state.budget
    self.tensor = obs

  def string_from(self, state, player):
    """Returns a string representation of the observation."""
    rstring = ""
    if player == 0:
      rstring = f"Sub:{state.sub_pos}, Timer:{state.timer:.1f}"
    elif player == 1:
      rstring = f"Heli:{state.heli_pos}, Timer:{state.timer:.1f}"
    return rstring


# registrera spelet i open_spiel
pyspiel.register_game(_GAME_TYPE, SubmarineHelicopterGame)