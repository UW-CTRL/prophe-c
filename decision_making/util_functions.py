import numpy as np
import heapq
import networkx as nx
from copy import deepcopy
import random as rd

class Node:
	def __init__(self, x, y, f=float('inf'), g=float('inf')):
		self.x = x
		self.y = y
		self.f = f
		self.g = g
		self.parent = None

	def __lt__(self, other):
		return self.f < other.f

DISTANCE_SCALER = 0.1


def shifted_relu(x, shift=0):
    return x if x > shift else -1


def kl_divergence(P, Q, scaler, node=None) -> float:
	"""
	Returns the KL-divergence of prior --> posterior.
	:param P: New distribution (posterior)
	:param Q: Reference distribution (prior)
	Uses log with base-2.
	"""
	return scaler * sum([P[var] * (np.log2(P[var]) / np.log2(Q[var])) for var in P.keys()])


def shannon_entropy(P) -> float:
    """
    Returns the Shannon entropy of the distribution P
    """
    return -1 * (sum([P[x] * np.log2(P[x]) for x in P.keys()]))


def max_likelihood(P, Q, scaler, node):
    return scaler * Q[node] * -1 # negative since we are technically minimizing cost


# TODO: investigate non-linear scaling between likelihood, entropy and costs (make high cost more desirable with higher certainty)
def combined(P, Q, scalers, node, next_node, graph):
    """
    Reward function that combines the probability of a node with the kl divergence between prior Q and posterior P
    """
    div_scale = scalers[0]
    likelihood_scale = scalers[1]
    
    div = sum([P[var] * (np.log2(P[var]) / np.log2(Q[var])) for var in P.keys()])
    mu = Q[next_node] * likelihood_scale
    
    k = Q[next_node] #/ (1 - Q[next_node])
    
    return (div * div_scale) + (mu * likelihood_scale)


def _euclid_distance(p1, p2, world=None) -> float:
	return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2), None


def obs_env_prep(G, seed=None, po=False):
    """
    Prepares an environment by remove certain nodes to simulate partial observability and randomly placing the
    agent at a starting position.
    """
    rooms = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "room"]
    if seed is not None:
        rd.seed(seed)
    
    starting_room = rd.sample(rooms, 1)[0]
    rd.seed()

    cpy_G = deepcopy(G)
    removed_nodes = []
    if po:
        # FULL PARTIAL OBSERVABILITY; NOT READY TO BE USED
        removed_nodes = [node for node in G.nodes() if node != starting_room and \
                          node not in G.neighbors(starting_room)]
    else:
        # PARTIAL OBSERVABILITY UNTIL THE LAYER ABOVE OBJECTS, CURRENTLY USED
        removed_nodes = [node for node in G.nodes() if nx.get_node_attributes(G, "class")[node] == "object"]
    
    cpy_G.remove_nodes_from(removed_nodes)
    return starting_room, cpy_G


def a_star(start, end, matrix):
    """
    A* implementation for a grid-world environment. Used primarily for animation of the planner's behavior.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    open_list = [(0, start)]
    came_from = {}
    g_score = {(row, col): float('inf') for row in range(rows) for col in range(cols)}
    g_score[start] = 0
    f_score = {(row, col): float('inf') for row in range(rows) for col in range(cols)}
    f_score[start] = _euclid_distance(start, end)[0]

    while open_list:
        _, current = heapq.heappop(open_list)

        if _euclid_distance(current, end)[0] < 3:
            path = []
            while current in came_from:
                path.insert(0, current)
                current = came_from[current]
            path.insert(0, start)
            return len(path), path

        for dr, dc in directions:
            neighbor = current[0] + dr, current[1] + dc

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and matrix[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1  # Assuming all moves have a cost of 1

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + _euclid_distance(neighbor, end)[0]
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found


def dijkstra_path(graph:nx.Graph, source, destination, po:bool, world=None, distance_func:callable=_euclid_distance) -> tuple:
    """
    Finds the shortest path between two points in an environment graph using Dijkstra's algorithm.
    Returns the length of the path and the list of nodes in the path, from start to finish.
    """
    if destination == source:
        return DISTANCE_SCALER, (source, source), 0
    
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0.0
    visited = set()
    previous_nodes = {}
    node_paths = {}

    while len(visited) < len(graph.nodes):
        # Find the node with the smallest distance from the source among unvisited nodes
        min_distance = float('inf')
        min_node = None
        for node in graph.nodes:
            if node not in visited and distances[node] < min_distance:
                min_distance = distances[node]
                min_node = node

        
        if min_node is None:
            break
        
        visited.add(min_node)
        
        # Update distances for neighbors of the current node
        for neighbor, attrs in graph[min_node].items():
            # aux_distance, _ = _euclid_distance(graph.nodes[neighbor]["centroid"], graph.nodes[min_node]["centroid"], world)
            centroid_distance = DISTANCE_SCALER
            # centroid_distance = 1
            total_distance = distances[min_node] + centroid_distance
            if total_distance <= distances[neighbor]:
                distances[neighbor] = total_distance
                previous_nodes[neighbor] = min_node
                # paths_seq.append()

    if destination not in previous_nodes:
        return float('inf'), (), float('inf')

    # Construct the shortest path
    path = (destination,)
    while path[-1] != source:
        path += (previous_nodes[path[-1]],)
    path = tuple(reversed(path))
    
    idx = 0
    path_len = len(path)
    euclid_distances = []
    
    if not po:
        while idx + 1 < path_len:
            # if graph.nodes[path[idx]].
            dist = _euclid_distance(graph.nodes[path[idx]]["centroid"], graph.nodes[path[idx+1]]["centroid"])[0]
            # dist =0
            euclid_distances.append(dist)
            idx += 1

    euclid_distance = sum(euclid_distances)

    return distances[destination], path, euclid_distance
