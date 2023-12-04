import networkx as nx
from world import house_rooms as hr
from world import house_objects as ho
from .util_functions import dijkstra_path
import spacy
import re
import numpy as np
import random as rd
from copy import deepcopy

SIMILARITY_SCALE = 10
MAX_ROOMS = 6
ROOM_CONNECTIONS = 2
TERMINAL_LEAF = "furniture"

# nlp = spacy.load('en_core_web_md')
# nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_core_web_lg')

def house_model_lite(graph:nx.Graph, estimation_variable:str, var_class:str="", class_hierarchy:list=[]):
	"""
	OLD AND LEGACY... NOT USING.
	Intended to provide priors in the case where the environment graph is fully known up to the layer
	of the estimation variable.
	"""
	prior = {}
	for node in graph.nodes():
		room_prob = 0.0
		furn_prob = 0.0

		if graph.nodes[node]["class"] == "room":
			obj_dict = getattr(hr, node[:-1]).object_associations
			if estimation_variable in obj_dict:
				room_prob = obj_dict[estimation_variable]
				obj_furniture_dict = getattr(ho, estimation_variable).furniture_associations
				if len(obj_furniture_dict) > 0:
					for edge_node in nx.neighbors(graph, node):
						if graph.nodes[edge_node]["class"] == "furniture":
							furn_dict = getattr(hr, node[:-1]).furniture_associations
							if edge_node[:-1] in furn_dict and edge_node[:-1] in obj_furniture_dict:
								furn_prob = obj_furniture_dict[edge_node[:-1]]
								prior[edge_node] = furn_prob * room_prob

				else:
					prior[node] = room_prob
	
	norm_const = sum(prior.values())
	for node in prior:
		prior[node] = prior[node] / norm_const

	return graph, prior


def query_priors_gen(G, query, root, tried:set):
	"""
	Constructs priors over the leaves of the graph given a query.
	Currently only considers furniture as leaves (TODO)
	"""
	# Look at leaves in graph to see if we have any new nodes/information 
	leaves = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "furniture" and node not in tried]

	rooms = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "room"]
	leaf_rooms = [node for node in rooms if not any(G.nodes[nbr]["class"] == "furniture" for nbr in G.neighbors(node))]

	leaves.extend(leaf_rooms)
	
	# Find the paths between the new node-leaves to their root; helps build the context-vector
	paths = [dijkstra_path(G, leaf, root, po=True)[1][:-1] for leaf in leaves]

	# Parse node labels into vectors -- these are "context vectors" for each decision node
	paths_copy = [["-".join(re.findall("[a-zA-Z]+", obj)) for obj in path] for path in paths]

	leaf_dict = {leaf: path for leaf, path in zip(leaves, paths_copy)}
	query = nlp(query)
	new_prior ={}
	
	# Lambda functions for computing the geometric mean and softmax of a generic vector
	geo_mean = lambda vec: np.power(np.product(vec), 1/len(vec))
	soft_max = lambda vec: np.exp(vec) / (sum(np.exp(vec)))

	prior = {leaf: geo_mean([np.abs(query.similarity(nlp(word))) * SIMILARITY_SCALE for word in leaf_dict[leaf]]) for leaf in leaf_dict}

	new_prior = {leaf: val for val, leaf in zip(soft_max(list(prior.values())), prior.keys())}
	return G, new_prior


def query_from_house_data(G, query, root, tried:set):
	"""
	Queries from the artificial probabilities given by the hand-made dataset.
	"""
	# Look at leaves in graph to see if we have any new nodes/information 
	leaves = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "furniture" and node not in tried]

	rooms = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "room"]
	leaf_rooms = [node for node in rooms if not any(G.nodes[nbr]["class"] == "furniture" for nbr in G.neighbors(node))]

	leaves.extend(leaf_rooms)

	prior = {}
	str_q = "_".join(re.findall("[a-zA-Z]+", query))
	furn_associations = getattr(ho, str_q).furniture_associations
	
	for leaf in leaves:
		if nx.get_node_attributes(G, "class")[leaf] == "furniture":
			if leaf[:-1] in furn_associations.keys():
				prior[leaf] = furn_associations[leaf[:-1]]
			else:
				prior[leaf] = 0

			for node in nx.neighbors(G, leaf):
				if nx.get_node_attributes(G, "class")[node] == "room":
					if str_q in getattr(hr, node[:-1]).object_associations.keys():
						prior[leaf] *= getattr(hr, node[:-1]).object_associations[str_q]
					else:
						prior[leaf] *= 0
				break
		
		elif nx.get_node_attributes(G, "class")[leaf] == "room":
			if str_q in getattr(hr, leaf[:-1]).object_associations.keys():
				prior[leaf] = getattr(hr, leaf[:-1]).object_associations[str_q] + np.random.uniform(-0.05, 0.1)
			else:
				prior[leaf] = 0
	
	 
	norm = sum(prior.values())
	prior_len = len(prior)

	if norm == 0:
		return G, {key: 1/prior_len for key in prior.keys()}
	
	prior = {key: val / norm for key, val in prior.items()}
	
	return G, prior	


def uniform_prior(G, query, root, tried:set):
	"""
	Returns a uniform distribution over the valid leaf-nodes in the graph
	"""
	leaves = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "furniture" and node not in tried]

	rooms = [node for node in G.nodes if nx.get_node_attributes(G, "class")[node] == "room"]
	leaf_rooms = [node for node in rooms if not any(G.nodes[nbr]["class"] == "furniture" for nbr in G.neighbors(node))]

	leaves.extend(leaf_rooms)
	len_leaves = len(leaves)
	return G, {leaf: 1.0 / len_leaves for leaf in leaves}


def complete_partial_graph(known_graph:nx.Graph):
	"""
	A simulated generative function that will 'fill out' the rest of a partially observed 3D scene graph by
	using some assumed rules for household environment generation. generates house up to the object layer.
	Ideally, this would be done in a more robust and general way, e.g. with a neural net.
	"""
	filled_graph = deepcopy(known_graph)

	rooms = [node for node in filled_graph.nodes if nx.get_node_attributes(filled_graph, "class")[node] == "room"]
	room_count_dict = {}
	
	for room in rooms:
		if room[:-1] in room_count_dict.keys():
			room_count_dict[room[:-1]] += 1
		else:
			room_count_dict[room[:-1]] = 1

	num_rooms = len(rooms)
	leaf_rooms = [room for room in rooms if len([node for node in nx.neighbors(filled_graph, room)]) == 2]
	num_leaves = len(leaf_rooms)
	num_added = 0
	power = lambda node, dict: 0 if node not in dict.keys() else dict[node]
	tries = 0
	
	
	# Add rooms
	while num_rooms + num_added < MAX_ROOMS and num_leaves > 0 and tries < 10:
		leaf = rd.choice(leaf_rooms)
		poss_adj_rooms = getattr(hr, leaf[:-1]).room_associations
		neighbor = rd.choices(population=list(poss_adj_rooms.keys()), weights=list(poss_adj_rooms.values()), k=1)[0]
		duplicate = getattr(hr, neighbor).duplicate
		
		if duplicate < 0.01 and neighbor[:-1] in room_count_dict.keys():
			print("here")
			tries += 1
			continue
		
		place = np.random.binomial(1, poss_adj_rooms[neighbor] * duplicate ** power(neighbor, room_count_dict))
		if place == 1:
			# Add new room to scene-graph
			new_room = neighbor + "9"
			i = 9
			while new_room in filled_graph.nodes():
				new_room = neighbor + str(i)
				i -= 1

			leaf_rooms.append(new_room)
			filled_graph.add_edge(leaf, new_room)
			filled_graph.add_edge("house", new_room)
			filled_graph.nodes[new_room]["class"] = "room"
			if neighbor not in room_count_dict.keys():
				room_count_dict[neighbor] = 1
			else:
				room_count_dict[neighbor] += 1

			# Possibly connect new room to other leaves
			num_additional_doorways = rd.randint(1, ROOM_CONNECTIONS+2)
			rel_rooms = [room for room in leaf_rooms if room[:-1] in getattr(hr, neighbor).room_associations.keys()]
			if len(rel_rooms) > 0:
				for i in range(num_additional_doorways):
					poss_edge = rd.choices(rel_rooms)[0]
					place_edge = np.random.binomial(1, getattr(hr, poss_edge[:-1]).room_associations[neighbor])
					if place_edge:
						filled_graph.add_edge(poss_edge, new_room)

			num_added += 1
		else:
			tries += 1 # stop bothering to place stuff after 10 tries

	# Add furniture
	furniture_nums = {}
	for leaf in leaf_rooms:
		room_f_dict = getattr(hr, leaf[:-1]).furniture_associations
		
		for f in room_f_dict.keys():
			prob = getattr(hr, leaf[:-1]).furniture_associations[f]
			placed = np.random.binomial(1, prob)
			if placed == 1:
				if f not in furniture_nums.keys():
					furniture_nums[f] = 1
				else:
					furniture_nums[f] += 1
				
				f_name = f + "9"
				
				i = 9
				while f_name in filled_graph.nodes():
					f_name = f+str(i - 1)
					i -= 1
				
				filled_graph.add_edge(leaf, f_name)
				filled_graph.nodes[f_name]["class"] = "furniture"
		
		# Fully connect the furniture for each room
		all_furniture = [node for node in nx.neighbors(filled_graph, leaf) if \
									  nx.get_node_attributes(filled_graph, "class")[node] == "furniture"]
		for child_node in all_furniture:
			for other_child in all_furniture:
					if child_node != other_child:
							filled_graph.add_edge(child_node, other_child)

	return filled_graph