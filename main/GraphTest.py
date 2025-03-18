"""
Filen hjälper för att manuellt kolla så att grafen är korrekt,
och så att heli_act_space och adjacency list är korrekta.
"""

import csv
import heapq
import copy

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
    

if __name__ == "__main__":
    obj = Graph("/Users/davidklasa/Documents/GitHub/Kexet/main/Test3.csv")
    for key in obj.nodes.keys():
        print(f"{key} : {obj.adjacency[key]}\n")

    print(f"################### separator ###################\n")

    for key in obj.nodes.keys():
        print(f"{key} : {obj.heli_act_space[key]}\n") 