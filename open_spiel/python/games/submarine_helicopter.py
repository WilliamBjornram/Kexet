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
import csv
import heapq

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

  def __init__(self, params=None):
    """konstruktor

    Args:
      params: (optional) dictionary av parametrar
      graph: en instans av graphClass
    """
    file = "/Users/davidklasa/Documents/KTH/KTH Kandidatexamensjobb/Kod/GymPPO/graph1.csv"
    self._graph =  Graph(file) # laddar in grafen
    self._budget = self._graph.calc_shortest_path() * 2
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

    super().__init__(_GAME_TYPE, _GAME_INFO, dict())

  def new_initial_state(self):
    """returnerar ett objekt med återställt state"""
    return SubmarineHelicopterState(self, self._graph, self._budget)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """returnerar ett objekt för observation"""
    return SubmarineHelicopterObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
        params,
        decay_factor=0.9)  # <-- you can pick any factor < 1.0)


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

    self.budget = budget
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

  ###### måste ta bort noder för heli där inte ubåt kan hittas så inte förgävesletar
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
      output.add(entry) if self.graph.discovery[entry] != 0 else None
      for x in tl:
        output.add(x) if self.graph.discovery[entry] != 0 else None
    return list(output)


class SubmarineHelicopterObserver:
  """Observer for the Submarine Helicopter game state.

  For simplicity, we build a flat observation vector consisting of:
    - One-hot encoding of the Sub's current node.
    - One-hot encoding of the Heli's current node.
    - A normalized timer value.
  """
  def __init__(self, iig_obs_type, params, decay_factor=0.9):
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Assume graph has N nodes. We will set N later when observing.
    self.tensor = None
    self.iig_obs_type = iig_obs_type
    self.decay_factor = decay_factor

  def set_from(self, state, player):
    N = len(state.graph.nodes)
    # We'll store:
    #   2*N for (Sub one-hot + Heli one-hot),
    #   +1 for the normalized timer,
    #   +N for the decayed visitation vector
    obs_size = 2 * N + 1 + N
    obs = np.zeros(obs_size, dtype=np.float32)
    if player == 0:
      # Player 0 sees its own (Sub's) position.
      obs[state.sub_pos] = 1.0  
      # Optionally, fill the opponent's part with zeros or a default value.
    elif player == 1:
      # Player 1 sees its own (Heli's) position.
      obs[N + state.heli_pos] = 1.0
    # Public info: the timer.
    obs[2*N] = state.timer/state.budget

    decayed_visits = np.zeros(N, dtype=np.float32)
    for (pl, action) in state.history:
        # Step A: Decay all existing visits (on *every* move)
        decayed_visits *= self.decay_factor

        # Step B: If it's the current player's move, increment that position
        if pl == player:
            decayed_visits[action] += 1.0

    # Place it at the end of the observation
    obs[2*N+1 : 2*N+1+N] = decayed_visits

    self.tensor = obs

    

  def string_from(self, state, player):
    # For demonstration, also show the decayed visits in string form
    # so you can visually debug.
    N = len(state.graph.nodes)
    decayed_visits = np.zeros(N, dtype=np.float32)
    for (pl, action) in state.history:
        decayed_visits *= self.decay_factor
        if pl == player:
            decayed_visits[action] += 1.0

    if player == 0:
        position_info = f"Sub pos: {state.sub_pos}"
    else:
        position_info = f"Heli pos: {state.heli_pos}"

    visits_info = f"Decayed visits: {decayed_visits}"
    return f"{position_info}, Timer: {state.timer:.1f}, {visits_info}"
  

#class for the graph the game is based of, loads graph from csv file
class Graph:
    def __init__(self, csv_file):
        # (x, y) position for each node save with node_id as key and (x, y) as tuple
        self.nodes = {}
        # dictionary for neighbors with node_id as key and neighbors as numpy array
        self.adjacency = {}
        # lists for start and end nodes for the sub
        self.start_nodes = []
        self.end_nodes = []
        # dictionary for keeping track of weights between nodes
        self.weights = {}
        # dictionary for probability of discovery
        self.discovery = {}

        # when initializing at end loads graph from csv file
        self.load_from_csv(csv_file)

    def load_from_csv(self, csv_file):
        # Expected columns: node_id:prob,x,y,is_start,is_end,neighbors:weights
        # Here we assume that after is_end, all subsequent fields are neighbors.
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
            size = len(rows)
            for row in rows:
                # for each row saves the values, see structure of csv file above
                node_id = int(row[0].split(":")[0])
                self.discovery[node_id] = int(row[0].split(":")[1])
                x = float(row[1])
                y = float(row[2])
                is_start = int(row[3])
                is_end = int(row[4])
                # remaining fields are neighbors:weights
                neighbors_w = [n for n in row[5:]]
                # empty list for neighbors
                neighbors = []
                for n in neighbors_w:
                    # first indice after split is node_id for neighbors
                    temp = n.split(":")
                    neighbors.append(int(temp[0]))
                    # creating a unique key for weights dictionary
                    key = str(node_id) + ":" + temp[0]
                    self.weights[key] = int(temp[1])

                # new entry into dictionaries
                self.nodes[node_id] = (x, y)
                self.adjacency[node_id] = neighbors

                self.start_nodes.append(node_id) if bool(is_start) else None
                self.end_nodes.append(node_id) if bool(is_end) else None

    # gör klassen iterable
    def __iter__(self):
        return iter(self.nodes)

    # för att kunna köra len(Graph)
    def __len__(self):
        return len(self.nodes)
    
    def calc_shortest_path(self):
      # Initialize distances and predecessors.
      dist = {node: float('inf') for node in self.nodes}
      prev = {node: None for node in self.nodes}

      # Use the lists as stored.
      start_nodes = self.start_nodes
      end_nodes = self.end_nodes

      if not start_nodes or not end_nodes:
          raise Exception("No start- or endnodes")

      # Multi-source initialization: set distance 0 for all start nodes.
      heap = []
      for s in start_nodes:
          dist[s] = 0
          heapq.heappush(heap, (0, s))

      # Run Dijkstra's algorithm.
      while heap:
          current_dist, u = heapq.heappop(heap)
          if current_dist > dist[u]:
              continue
          # Directly iterate over the list of neighbors.
          for v in self.adjacency[u]:
              key = f"{u}:{v}"
              # Get the edge weight; if missing, skip this neighbor.
              weight_uv = self.weights.get(key)
              if weight_uv is None:
                  continue
              alt = current_dist + weight_uv
              if alt < dist[v]:
                  dist[v] = alt
                  prev[v] = u
                  heapq.heappush(heap, (alt, v))

      # Among all end nodes, pick the one with the smallest distance.
      best_end = None
      best_cost = float('inf')
      for e in end_nodes:
          if dist[e] < best_cost:
              best_cost = dist[e]
              best_end = e

      if best_end is None or best_cost == float('inf'):
          raise Exception("No shortest path found.")

      return best_cost

# registrera spelet i open_spiel
pyspiel.register_game(_GAME_TYPE, SubmarineHelicopterGame)
