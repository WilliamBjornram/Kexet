import csv
import numpy as np
import heapq
#class for the graph the game is based of, loads graph from csv file

class Graph:
    def __init__(self, csv_file):
        # (x, y) position for each node save with node_id as key and (x, y) as tuple
        self.nodes = {}
        # dictionary for neighbors with node_id as key and neighbors as numpy array
        self.adjacency = {}
        # lists for start and end nodes for the sub
        self.start_nodes = {}
        self.end_nodes = {}
        # dictionary for keeping track of weights between nodes
        self.weights = {}
        # dictionary for probability of discovery
        self.discovery = {}

        # when initializing at end loads graph from csv file
        self.load_from_csv(csv_file)
        temp = self.calc_shortest_path()
        print(temp)

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
                self.adjacency[node_id] = self.convert_to_mask(neighbors, size)

                self.start_nodes[node_id] = bool(is_start)
                self.end_nodes[node_id] = bool(is_end)

    def neighbors(self, node):
        return self.adjacency[node]
    
    def convert_to_mask(self, neighbors, size):
        # initialiserar en array med samma size som action space
        mask = np.zeros(size, dtype=bool)
        # sätter alla legala drag till True
        mask[neighbors] = True
        return mask
    
    def calc_shortest_path(self):

        # Initialize distances and predecessors.
        dist = {node: float('inf') for node in self.nodes}
        prev = {node: None for node in self.nodes}

        # Get all start and end nodes.
        start_nodes = [node for node, flag in self.start_nodes.items() if flag]
        end_nodes = [node for node, flag in self.end_nodes.items() if flag]

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
            # Retrieve neighbors using the boolean mask in self.adjacency[u].
            # np.where returns a tuple with an array; we take the first element.
            for v in np.where(self.adjacency[u])[0]:
                v = int(v)  # ensure v is a plain int
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
