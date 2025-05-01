"""Submarine Helicopter game in Python using OpenSpiel.

The game is zero-sum and is an imperfect information game.
Played on a graph designed to simulate an archipelago environment.
Player 0 (Sub) moves along neighboring nodes and has a move budget.
Player 1 (Heli) moves one or two neighbors away.
"""

import numpy as np
import pyspiel
import math
import csv
import heapq
import copy

# Player 0 == Sub, Player 1 == Helicopter
_NUM_PLAYERS = 2

_DEFAULT_PARAMS = {
    "filename": "/Users/davidklasa/Documents/GitHub/Kexet/main/grafer/Graf2.csv"
}

_GAME_TYPE = pyspiel.GameType(
    short_name="python_submarine_helicopter",
    long_name="Python Submarine Helicopter",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification=_DEFAULT_PARAMS)

class SubmarineHelicopterGame(pyspiel.Game):
  """A Python version of the Submarine Helicopter game using OpenSpiel."""

  def __init__(self, params=_DEFAULT_PARAMS):
    """constructor
    Args:
      params: dictionary of parameters
    """
    file = params["filename"]
    self._graph =  Graph(file) # loads the graph
    self._budget = self._graph.calc_shortest_path() * 2
    max_moves = math.ceil(self._budget/10) # takes budget/10 and rounds up to get max number of moves
                                    # becomes a worst-case scenario where the sub only makes moves without progress
    
    # sets action space to all nodes, we later provide the relevant subset per node
    max_act = len(self._graph.nodes)

    _GAME_INFO = pyspiel.GameInfo(
        num_distinct_actions=max_act, # varies, so we set to max number of actions
        max_chance_outcomes=2, # when sub.pos == heli.pos chance gives two alternatives
        num_players=_NUM_PLAYERS,
        min_utility=-1.0, # reward max, min and sum (zero sums)
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=max_moves)
    
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """returns an object with reset state"""
    return SubmarineHelicopterState(self, self._graph, self._budget)

  def make_py_observer(self, iig_obs_type=None, params=_DEFAULT_PARAMS):
    """returns an observation object"""
    return SubmarineHelicopterObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
        params,
        decay_factor=0.9) # we use this factor so the algorithm sees the history of where one has been 


class SubmarineHelicopterState(pyspiel.State):
  """
  State keeps track of:
    - sub_pos: node_id for the submarine's current position
    - heli_pos: node_id for the helicopter's current position
    - timer: remaining budget for the submarine
    - _current_player: whose turn it is
    - _game_over: flag indicating whether the game is over
  """

  def __init__(self, game, graph, budget):
    """initializes the game"""
    super().__init__(game)
    self.graph = graph

    self.budget = budget
    self.timer = budget
    
    # starting positions != random per CFR, hence start at specific points
    self.sub_pos = 0
    self.heli_pos = 4

    self._game_over = False

    # the submarine moves first
    self._current_player = 0

    # to track which moves have been made
    self.history = []

    # flag needed for CFR chance event
    self._pending_chance_event = False

  def information_state_string(self, player=None):
      """
      Returns a string representation of the information state for the given player.
      String includes only information available to that player,
      including its history of information, weighted by when it occurred.
      """
      if player is None:
        player = self.current_player()
      # Normalizes timer value
      normalized_timer = self.timer / self.budget
      decay_factor = 0.9  # Decay factor for history
      
      # Calculates decayed visits for the history, gives: [pos1*0.9^2, pos2*0.9, pos3], at the third node
      decayed_visits = np.zeros(len(self.graph.nodes), dtype=np.float32)
      for (pl, action) in self.history:
          decayed_visits *= decay_factor
          if pl == player:
              decayed_visits[action] += 1.0
      
      if player == 0:
          return f"SubPos:{self.sub_pos}|Timer:{normalized_timer:.2f}|SubDecayedVisits:{decayed_visits.tolist()}"
      elif player == 1:
          return f"HeliPos:{self.heli_pos}|Timer:{normalized_timer:.2f}|HeliDecayedVisits:{decayed_visits.tolist()}"
      else:
          return str(self.history)
      
  
  def information_state_tensor(self, player=None):
    """
    Returns a tensor describing the state for the current player, imperfect information
    so has no information about the opponent. See information_state_string for details, same setup.
    Implements the same functionality as the set_from function in observer.
    """
    if player is None:
      player = self.current_player()

    N = len(self.graph)
    obs_size = 4 * N + 1
    decay_factor = 0.9
    tensor = np.zeros(obs_size, dtype=np.float32)
    # normalized value for timer
    tensor[-1] = self.timer/self.budget

    decayed_visits = np.zeros(N, dtype=np.float32)
    for (pl, action) in self.history:
        # multiply vector by decay_factor
        decayed_visits *= decay_factor
        # increment the position it has been in
        if pl == player:
            decayed_visits[action] += 1.0

    # player only sees their own position
    if player == 0:
      tensor[self.sub_pos] = 1.0 
      tensor[N:2*N] = -1
      tensor[2*N : 3*N] = decayed_visits
      tensor[3*N:4*N] = -1
    elif player == 1:
      tensor[N + self.heli_pos] = 1.0
      tensor[0:N] = -1
      tensor[2*N:3*N] = -1
      tensor[3*N : 4*N] = decayed_visits
    
    return tensor
  
  # CFR needs clone function
  def clone(self):
    """creates a new state that is a complete copy"""
    new_state = SubmarineHelicopterState(self.get_game(), self.graph, self.budget)
    new_state.timer = self.timer
    new_state.sub_pos = self.sub_pos
    new_state.heli_pos = self.heli_pos
    new_state._game_over = self._game_over
    new_state._current_player = self._current_player
    new_state.history = list(self.history)
    return new_state

  def current_player(self):
    """returns the id of the current player or terminal if game is over"""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    # if ongoing chance event return chance
    if self._pending_chance_event:
      return pyspiel.PlayerId.CHANCE
    return self._current_player

  def _legal_actions(self, player):
    """returns a list of legal moves for the current player"""
    # chance nodes are treated as extra players  
    if player == pyspiel.PlayerId.CHANCE:
      # 0 means detection and 1 means no detection
      return [0, 1]
    if player == 0:
      return self.graph.adjacency[self.sub_pos] # moves to an adjacent node
    elif player == 1:
      # all adjacent nodes except end nodes  
      return [node for node in self.graph.adjacency[self.heli_pos] if node not in self.graph.end_nodes]
    else:
      return []

  def chance_outcomes(self):
    """function is called when a chance event is ongoing"""
    if self.sub_pos == self.heli_pos and self._pending_chance_event:
      # chance of detection is a number 0 to 10
      p_detection = self.graph.discovery[self.sub_pos] / 10.0
      return [(0, p_detection), (1, 1 - p_detection)]
    return []

  def _apply_action(self, action):
    """executes action"""
    # handles chance node separately
    if self.current_player() == pyspiel.PlayerId.CHANCE:
      if action == 0:  # detection -> terminal
        self._game_over = True
        self._returns = [-1, 1]
      elif action == 1:  # no detection: remove chance event and continue
        self._pending_chance_event = False

        # check if episode is done
        terminal, reward = self._check_terminal()
        if terminal:
          self._game_over = True
          # if done, return reward
          self._returns = [reward, -reward]
          return

        # if last move was by sub -> heli's turn, and vice versa
        if self.history[-1][0] == 0:
          self._current_player = 1
        else:
          self._current_player = 0
      return

    # keep track of moves made
    self.history.append((self._current_player, action))

    if self._current_player == 0:
      # if it's the submarine's move
      key = f"{self.sub_pos}:{action}"
      move_cost = self.graph.weights.get(key, float('inf'))
      self.timer -= move_cost
      if action not in self._legal_actions(self._current_player):
        raise Exception("Illegal move from Submarine.")
      self.sub_pos = action

      # If the sub moves into the heli's position, trigger chance event.
      if self.sub_pos == self.heli_pos:
        self._pending_chance_event = True
        self._current_player = pyspiel.PlayerId.CHANCE
        return

      # check if the game is over
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        # if done, return reward
        self._returns = [reward, -reward]
        return

      # switch turn to next player
      self._current_player = 1
      
    elif self._current_player == 1:
      # check for legal move
      if action not in self._legal_actions(self._current_player):
        raise Exception("Illegal move from Helicopter.")
      self.heli_pos = action

      # If the heli moves into the sub's position, trigger chance event.
      if self.sub_pos == self.heli_pos:
        self._pending_chance_event = True
        self._current_player = pyspiel.PlayerId.CHANCE
        return

      # same as above
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        self._returns = [reward, -reward]
        return
      
      # switch player
      self._current_player = 0

  def _check_terminal(self):
    """checks if the episode is over and returns (terminal_flag, reward).
    reward is from the submarine's perspective (and the game is zero-sum, so the helicopter's is the inverse).
    terminal for chance event is handled separately.
    """
    # if sub is at end node -> game over
    if self.sub_pos in self.graph.end_nodes:
      return True, +1
    # if timer runs out -> negative reward
    if self.timer <= 0:
      return True, -1
    return False, 0

  def returns(self):
    """returns reward if state is terminal,
    otherwise if not terminal -> 0
    """
    if not self._game_over:
      return [0.0, 0.0]
    return self._returns
  
  def is_terminal(self):
    """returns True if the game is over (mandatory function)"""
    return self._game_over

  def __str__(self):
    """returns a string representation of the state"""
    return (f"Sub: {self.sub_pos}, Heli: {self.heli_pos}, "
            f"Timer: {self.timer:.1f}, History: {self.history}")


class SubmarineHelicopterObserver:
  """Observer for game state.

  For simplicity, we build a flat observation vector consisting of:
    - One-hot encoding of Sub's pos.
    - One-hot encoding of Heli's pos.
    - History of position for Sub's pos, decayed visits
    - History of position for Heli's pos, decayed visits
    - A normalized timer value (between 0 and 1)
  """
  def __init__(self, iig_obs_type, params, decay_factor=0.9):
    if params:
      raise ValueError(f"Observation parameter not supported; got {params}")

    self.tensor = None
    self.dict = None
    self.iig_obs_type = iig_obs_type
    self.decay_factor = decay_factor

  def set_from(self, state, player=None):

    if player == None:
      player = state.current_player()

    N = len(state.graph)
    obs_size = 4 * N + 1
    obs = np.zeros(obs_size, dtype=np.float32)
    # normalized value for timer
    obs[-1] = state.timer/state.budget

    decayed_visits = np.zeros(N, dtype=np.float32)
    for (pl, action) in state.history:
        # multiply vector by decay_factor
        decayed_visits *= self.decay_factor
        # increment the position it has been in
        if pl == player:
            decayed_visits[action] += 1.0

    # place the vector where it should be
    if player == 0:
      obs[state.sub_pos] = 1.0 
      obs[N:2*N] = -1
      obs[2*N:3*N] = decayed_visits
      obs[3*N:4*N] = -1
    elif player == 1:
      obs[0:N] = -1
      obs[N + state.heli_pos] = 1.0
      obs[2*N:3*N] = -1
      obs[3*N:4*N] = decayed_visits
    
    self.tensor = obs
    self.dict = {"observation": obs.tolist()}

  def string_from(self, state, player):
    """prints observation"""
    N = len(state.graph)
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
  

# class for the graph, loads the graph from csv file
class Graph:
  def __init__(self, csv_file):
      # (x, y) position for each node saved with node_id as key and (x, y) as tuple
      self.nodes = {}
      # action space for submarine
      self.adjacency = {}
      # action space for heli
      self.heli_act_space = {}
      # lists for start and end nodes for the submarine
      self.start_nodes = []
      self.end_nodes = []
      # dictionary to keep track of transition weights
      self.weights = {}
      # dictionary for detection probability
      self.discovery = {}

      # loads the graph
      self.load_from_csv(csv_file)

  def load_from_csv(self, csv_file):
    """expects columns: node_id:prob,x,y,is_start,is_end,neighbors:weights
    after is_end, all remaining entries are neighbors"""
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        size = len(rows)
        for row in rows:
            # for each row save values in lists/dictionary 
            node_id = int(row[0].split(":")[0])
            self.discovery[node_id] = int(row[0].split(":")[1])
            x = float(row[1])
            y = float(row[2])
            is_start = int(row[3])
            is_end = int(row[4])
            # remaining are neighbors:weights
            neighbors_w = [n for n in row[5:]]
            # empty list for neighbors
            neighbors = []
            for n in neighbors_w:
                # first index after split is node_id for neighbors
                temp = n.split(":")
                neighbors.append(int(temp[0]))
                # creates unique key for weights dictionary
                key = str(node_id) + ":" + temp[0]
                self.weights[key] = int(temp[1])

            # new entries to dictionaries
            self.nodes[node_id] = (x, y)
            self.adjacency[node_id] = neighbors

            self.start_nodes.append(node_id) if bool(is_start) else None
            self.end_nodes.append(node_id) if bool(is_end) else None

    # heli should be able to move freely and not just forward
    for key in self.adjacency:
      temp = set()
      for ind in self.adjacency:
        for node in self.adjacency[ind]:
          temp.add(ind) if node == key and ind not in self.end_nodes else None
          temp.add(node) if ind == key and node not in self.end_nodes else None
      self.heli_act_space[key] = list(temp)
    
    # check for existence of start and end nodes
    if not self.start_nodes:
      raise Exception("No start nodes defined in the graph.")
    elif not self.end_nodes:
        raise Exception("No end nodes defined in the graph.")
    
    if len(self.nodes) != len(self.adjacency) or len(self.nodes) != len(self.discovery):
        raise Exception("Dimensions for dictionaries do not match.")

  # make the class iterable
  def __iter__(self):
    return iter(self.nodes)

  # to enable len(Graph)
  def __len__(self):
    return len(self.nodes)
  
  def calc_shortest_path(self):
    # initialize distances and predecessors
    dist = {node: float('inf') for node in self.nodes}
    prev = {node: None for node in self.nodes}

    # use the lists
    start_nodes = self.start_nodes
    end_nodes = self.end_nodes

    # set distance zero for all start nodes
    heap = []
    for s in start_nodes:
        dist[s] = 0
        heapq.heappush(heap, (0, s))

    # run Dijkstra's algorithm
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        # iterate over list of neighbors
        for v in self.adjacency[u]:
            key = f"{u}:{v}"
            # get the weight for the transition
            weight_uv = self.weights.get(key)
            if weight_uv is None:
                continue
            alt = current_dist + weight_uv
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    # choose node with shortest distance
    best_end = None
    best_cost = float('inf')
    for e in end_nodes:
        if dist[e] < best_cost:
            best_cost = dist[e]
            best_end = e

    if best_end is None or best_cost == float('inf'):
        raise Exception("Did not find a shortest path to end node.")

    return best_cost

# register the game in open_spiel
pyspiel.register_game(_GAME_TYPE, SubmarineHelicopterGame)
