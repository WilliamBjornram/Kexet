"""Submarine Helicopter game i Python med OpenSpiel.

Spelet är zero-sum och är ett imperfekt informations spel.
Spelas på en graf tänkt att simulera en skärgårdsmiljö.
Player 0 (Sub) rör sig längs grann-noder och har en budget för antalet drag.
Player 1 (Heli) rör sig en eller två grannar bort.
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
    "filename": "/content/Kexet/main/grafer/Test0.csv"
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
  """en Python version av spelet Submarine Helicopter mha OpenSpiel."""

  def __init__(self, params=_DEFAULT_PARAMS):
    """konstruktor
    Args:
      params: (optional) dictionary av parametrar
    """
    file = params["filename"]
    self._graph =  Graph(file) # laddar in grafen
    self._budget = self._graph.calc_shortest_path() * 2
    max_moves = math.ceil(self._budget/10) # tar budget/10 och rundar uppåt för att få max antal drag
                                    # blir ett worst case scenario där ubåt bara gör drag men inte kommer någon vart
    
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
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True),
        params,
        decay_factor=0.9) # vi har denna faktor för att algortimen ska se historien av vart man varit 


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
    
    # startpositioner != random enligt CFR, därav start i bestämda punkter
    self.sub_pos = 0
    self.heli_pos = self.graph.end_nodes[0]

    self._game_over = False

    # ubåten rör sig först
    self._current_player = 0

    # för att hålla koll på vilka drag som gjorts
    self.history = []

    # behövs flagga för CFR chance event
    self._pending_chance_event = False

  def information_state_string(self, player=None):
      """
      Returnerar en string representation av information state för den givna spelaren.
      String inkluderar enbart information som finns tillgänglig till den spelaren,
      bland annat dess historik av information, avräknat för när den var där.
      """
      if player is None:
        player = self.current_player()
      # Normaliserar timer värde
      normalized_timer = self.timer / self.budget
      decay_factor = 0.9  # Decay factor för historik
      
      # Beräknar decayed besök för historiken, ger: [pos1*0,9^2, pos2*0,9, pos3], vid tredje noden
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
    Returnerar en tensor som beskriver state för aktuell spelare, imperfect information
    så har ingen information om motspelaren. Se information_state_string för vidare info, samma upplägg.
    Implementerar samma funktionalitet som funktionen set_from i observer.
    """
    if player is None:
      player = self.current_player()

    N = len(self.graph)
    obs_size = 4 * N + 1
    decay_factor = 0.9
    tensor = np.zeros(obs_size, dtype=np.float32)
    # player ser bara sin egna position
    if player == 0:
      tensor[self.sub_pos] = 1.0 
    elif player == 1:
      tensor[N + self.heli_pos] = 1.0
    # normaliserat värde för timer
    tensor[-1] = self.timer/self.budget

    decayed_visits = np.zeros(N, dtype=np.float32)
    for (pl, action) in self.history:
        # gångra vectorn med decay_factor
        decayed_visits *= decay_factor
        # inkrementera positionen där den varit
        if pl == player:
            decayed_visits[action] += 1.0

    # placera in vectorn där den ska vara
    if player == 0:
      tensor[2*N : 3*N] = decayed_visits
    elif player == 1:
      tensor[3*N : 4*N] = decayed_visits
    
    return tensor
  
  # CFR behöver clone function
  def clone(self):
    """skapar ett nytt state som är en komplett kopia"""
    new_state = SubmarineHelicopterState(self.get_game(), self.graph, self.budget)
    new_state.timer = self.timer
    new_state.sub_pos = self.sub_pos
    new_state.heli_pos = self.heli_pos
    new_state._game_over = self._game_over
    new_state._current_player = self._current_player
    new_state.history = list(self.history)
    return new_state

  def current_player(self):
    """returnerar id av den aktuella spelaren annars om spel slut -> terminal"""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    # om pågående chance event returnerar chance
    if self._pending_chance_event:
      return pyspiel.PlayerId.CHANCE
    return self._current_player

  def _legal_actions(self, player):
    """returnerar en lista på legala drag för aktuell spelare"""
    # chance noder behandlas som extra spelare  
    if player == pyspiel.PlayerId.CHANCE:
      # 0 betyder detektion och 1 betyder ingen detektion
      return [0, 1]
    if player == 0:
      return self.graph.adjacency[self.sub_pos] # rör sig till någon adjecent nod
    elif player == 1:
      return self.graph.heli_act_space[self.heli_pos] # dictionary som har alla legal moves som lista i en dictionary över alla noder
    else:
      return []

  def chance_outcomes(self):
    """funktionen kallas när vi har ett chance event pågående"""
    if self.sub_pos == self.heli_pos and self._pending_chance_event:
      # chance av detektion är ett tal 0 till 10
      p_detection = self.graph.discovery[self.sub_pos] / 10.0
      return [(0, p_detection), (1, 1 - p_detection)]
    return []

  def _apply_action(self, action):
    """genomför action"""
    # hanterar chance node separat
    if self.current_player() == pyspiel.PlayerId.CHANCE:
      if action == 0:  # detektion -> terminal
        self._game_over = True
        self._returns = [-1, 1]
      elif action == 1:  # ingen detektion: ta bort chance event och fortsätt
        self._pending_chance_event = False

        # kollar ifall episode slut
        terminal, reward = self._check_terminal()
        if terminal:
          self._game_over = True
          # om klart så returnerar vi reward
          self._returns = [reward, -reward]
          return

        # om senaste drag gjordes av ubåt -> helis tur och tvärtom
        if self.history[-1][0] == 0:
          self._current_player = 1
        else:
          self._current_player = 0
      return

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

      # If the sub moves into the heli's position, trigger chance event.
      if self.sub_pos == self.heli_pos:
        self._pending_chance_event = True
        self._current_player = pyspiel.PlayerId.CHANCE
        return

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

      # If the heli moves into the sub's position, trigger chance event.
      if self.sub_pos == self.heli_pos:
        self._pending_chance_event = True
        self._current_player = pyspiel.PlayerId.CHANCE
        return

      # samma som ovan
      terminal, reward = self._check_terminal()
      if terminal:
        self._game_over = True
        self._returns = [reward, -reward]
        return
      
      # byter spelare
      self._current_player = 0

  def _check_terminal(self):
    """kollar om episoden är slut och returnerar (terminal_flag, belöning).
    belöning är från ubåtens perspektiv (och spelet är zero-sum, så omvända belöningen är helikopterns).
    terminal för chance event hanteras separat.
    """
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
    """returnerar True om spelet är över (obligatorisk funktion)"""
    return self._game_over

  def __str__(self):
    """returnerar en string som representation över state"""
    return (f"Sub: {self.sub_pos}, Heli: {self.heli_pos}, "
            f"Timer: {self.timer:.1f}, History: {self.history}")


class SubmarineHelicopterObserver:
  """Observer för game state.

  För enkelhetens skull bygger vi en platt observations vektor bestående av:
    - One-hot encoding av Sub's pos.
    - One-hot encoding av Heli's pos.
    - Historia av position för Sub's pos, decayed visits
    - Historia av position för Heli's pos, decayed visits
    - En normaliserad timer värde (mellan 0 och 1)
  """
  def __init__(self, iig_obs_type, params, decay_factor=0.9):
    if params:
      raise ValueError(f"Observation parameter stöttas ej; fick {params}")

    self.tensor = None
    self.dict = None
    self.iig_obs_type = iig_obs_type
    self.decay_factor = decay_factor

  def set_from(self, state, player):
    N = len(state.graph)
    obs_size = 4 * N + 1
    obs = np.zeros(obs_size, dtype=np.float32)
    # player ser bara sin egna position
    if player == 0:
      obs[state.sub_pos] = 1.0 
    elif player == 1:
      obs[N + state.heli_pos] = 1.0
    # normaliserat värde för timer
    obs[-1] = state.timer/state.budget

    decayed_visits = np.zeros(N, dtype=np.float32)
    for (pl, action) in state.history:
        # gångra vectorn med decay_factor
        decayed_visits *= self.decay_factor
        # inkrementera positionen där den varit
        if pl == player:
            decayed_visits[action] += 1.0

    # placera in vectorn där den ska vara
    if player == 0:
      obs[2*N : 3*N] = decayed_visits
    elif player == 1:
      obs[3*N : 4*N] = decayed_visits

    self.tensor = obs
    self.dict = {"observation": obs.tolist()}

  def string_from(self, state, player):
    """skriver ut observation"""
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
  

#class för grafen, laddar grafen från csv fil
class Graph:
  def __init__(self, csv_file):
      # (x, y) position för varje nod sparas som node_id som nyckel och (x, y) som tuple
      self.nodes = {}
      # dictionary för grannar med node_id som nyckel och grannar som lista
      self.adjacency = {}
      # listor för start och slutnoder för ubåten
      self.start_nodes = []
      self.end_nodes = []
      # dictionary för att hålla koll på vikter mellan övergångar
      self.weights = {}
      # dictionary för sannolikhet av upptäckt
      self.discovery = {}
      # dictionary för heli's möjliga drag i varje position
      self.heli_act_space = {}

      # laddar in grafen
      self.load_from_csv(csv_file)
      # kallar funktionen som sätter dictionary för heli's legal moves
      self.heli_act()

  def load_from_csv(self, csv_file):
    """förväntar sig kolumnerna: node_id:prob,x,y,is_start,is_end,neighbors:weights
    efter is_end, alla efter det är grannar"""
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
        size = len(rows)
        for row in rows:
            # för varje rad sparar värdena i listor/dictionary 
            node_id = int(row[0].split(":")[0])
            self.discovery[node_id] = int(row[0].split(":")[1])
            x = float(row[1])
            y = float(row[2])
            is_start = int(row[3])
            is_end = int(row[4])
            # resterande är neighbors:weights
            neighbors_w = [n for n in row[5:]]
            # tom lista för grannar
            neighbors = []
            for n in neighbors_w:
                # första index efter split är node_id for grannar
                temp = n.split(":")
                neighbors.append(int(temp[0]))
                # skapar unik nyckel för weights dictionary
                key = str(node_id) + ":" + temp[0]
                self.weights[key] = int(temp[1])

            # nya entries till dictionaries
            self.nodes[node_id] = (x, y)
            self.adjacency[node_id] = neighbors

            self.start_nodes.append(node_id) if bool(is_start) else None
            self.end_nodes.append(node_id) if bool(is_end) else None

    # kontrollerar så finns start och slutnoder
    if not self.start_nodes:
      raise Exception("Inga startnoder definerade i grafen.")
    elif not self.end_nodes:
        raise Exception("Inga slutnoder definierade i grafen.")
    
    if len(self.nodes) != len(self.adjacency) or len(self.nodes) != len(self.discovery):
        raise Exception("Dimensioner för dictionaries stämmer ej.")

  # gör klassen iterable
  def __iter__(self):
    return iter(self.nodes)

  # för att kunna köra len(Graph)
  def __len__(self):
    return len(self.nodes)
  
  def heli_act(self):
    """blir en unik lista med alla neighbors och indirekta neighbors (två steg)
    tar bort noder med discovery rate 0 (transit noder)
    """
    temp = copy.deepcopy(self)
    # adderar så att adjacency list för varje nod motsvarar varandra
    for k in temp.adjacency.keys():
        tl = temp.adjacency[k]
        for i in tl:
          if k not in temp.adjacency[i]:
              temp.adjacency[i].append(k)

    N = 2 # kan röra sig upp till två noder bort varje tidssteg
    for key in self.nodes.keys():
      output = help_heli_act(temp, N, key)
      self.heli_act_space[key] = list(output)
  
  def calc_shortest_path(self):
    # initialiserar distanser och föregångare
    dist = {node: float('inf') for node in self.nodes}
    prev = {node: None for node in self.nodes}

    # använd listorna
    start_nodes = self.start_nodes
    end_nodes = self.end_nodes

    # sätter distans noll för alla startnoder
    heap = []
    for s in start_nodes:
        dist[s] = 0
        heapq.heappush(heap, (0, s))

    # kör Dijkstra's algoritm
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > dist[u]:
            continue
        # iterar över listan av grannar
        for v in self.adjacency[u]:
            key = f"{u}:{v}"
            # få vikten för övergången
            weight_uv = self.weights.get(key)
            if weight_uv is None:
                continue
            alt = current_dist + weight_uv
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    # välj noden med kortast distans
    best_end = None
    best_cost = float('inf')
    for e in end_nodes:
        if dist[e] < best_cost:
            best_cost = dist[e]
            best_end = e

    if best_end is None or best_cost == float('inf'):
        raise Exception("Fann ingen kortaste väg till slutnod.")

    return best_cost

def help_heli_act(graph, N, key):
    """beräknar rekursivt de noder som kan nås från `key` inom N steg.
    
    Enbart chokepoint nodes inkluderas
    
    Args:
      graph: grafen (objekt) som det ska beräknas på
      N: antal steg bort (rekursivt djup)
      key: startnoden
    
    Returns:
      Ett antal noder som går att nå N steg bort (inklusive startnoden)
    """
    result = set()
    if N == 0:
        # basfallet, om inte transit node och nått rekursivt djup inkludera noden
        if graph.discovery[key] != 0:
            result.add(key)
        return result
    else:
        # om ej nått rekursivt djup, uppdatera då resultatet för varje granne
        for neighbor in graph.adjacency[key]:
            result.update(help_heli_act(graph, N-1, neighbor))
        # addera eventuellt den nuvarande noden också
        if graph.discovery[key] != 0:
            result.add(key)
        return result

# registrera spelet i open_spiel
pyspiel.register_game(_GAME_TYPE, SubmarineHelicopterGame)
