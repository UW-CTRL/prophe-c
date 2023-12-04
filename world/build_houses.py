from .house_gen import House
# from house_gen import House
from . import house_rooms, house_objects
# import house_objects, house_rooms
# from .house_objects import *
import plotly.graph_objects as go
import networkx as nx
import pickle
import logging
import ast
import dataclasses

class HouseBuilder:
	def __init__(self) -> None:
		pass

	def build_house_files(self, num_houses, file_dir):
		num_times = 863
		while num_times < num_houses:
			try:
				h=House()
				vals = h.gen()
				if vals is not None:
					scene_graph_list = nx.to_dict_of_lists(vals[3])
					meta_data = vals[2]
					img = vals[1]
					storage_dict = {"world": img, "data": meta_data, "scene_graph": scene_graph_list}
					pickle.dump(storage_dict, open(file_dir + f'/house{num_times}.pickle', 'wb'))
					num_times += 1
					print("FINISHED")
				else:
					logging.warning("Regenerating...")
			except Exception as e:
				pass

	def visualize_graph(self, meta_data, graph):
		coords = {}
		colors = {}

		for node in meta_data.keys():
				x_y = (meta_data[node]["centroid"][1], meta_data[node]["centroid"][0])
				if meta_data[node]["class"] == "building":
						coords[meta_data[node]["name"]] = x_y + (600,)
						colors[meta_data[node]["name"]] = 1
				elif meta_data[node]["class"] == "room":
						coords[meta_data[node]["name"]] = x_y + (500,)
						colors[meta_data[node]["name"]] = 2
				elif meta_data[node]["class"] == "furniture":
						coords[meta_data[node]["name"]] = x_y + (200,)
						colors[meta_data[node]["name"]] = 3
				elif meta_data[node]["class"] == "object":
						coords[meta_data[node]["name"]] = x_y + (0,)
						colors[meta_data[node]["name"]] = 4
		# Extract the coordinates of the nodes for plotting
		X, Y, Z = zip(*coords.values())

		# Create a Scatter plot for nodes
		node_trace = go.Scatter3d(
				x=X,
				y=Y,
				z=Z,
				mode='markers+text',
				text=[f"{node}" for node in coords.keys()],  # Labels for all nodes
				textposition='bottom center',  # Position of the labels
				textfont=dict(size=25),
				marker=dict(size=12)
		)

		node_colors = [colors[node] for node in colors.keys()]


		# Create an edge trace for lines connecting the nodes
		edge_x = []
		edge_y = []
		edge_z = []

		for edge in graph.edges():
				# room_name = meta_data[]
				node1_idx, node2_idx = edge
				x0, y0, z0 = coords[node1_idx]
				x1, y1, z1 = coords[node2_idx]
				edge_x.extend([x0, x1, None])
				edge_y.extend([y0, y1, None])
				edge_z.extend([z0, z1, None])

		edge_trace = go.Scatter3d(
				x=edge_x,
				y=edge_y,
				z=edge_z,
				mode='lines',
				line=dict(width=3.0, color='black')
				)

		axis = dict(showbackground=False,
						showline=False,
						zeroline=False,
						showgrid=False,
						showticklabels=False,
						title=''
						)


		layout = go.Layout(title="Scene-graph", showlegend=False, width=1000, height=1000, scene=dict(
							xaxis=dict(axis),
							yaxis=dict(axis),
							zaxis=dict(axis),
						),
			margin=dict(
				t=100
		), )

		fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
		fig.update_traces(marker=dict(color=node_colors, colorscale='Rainbow', cmin=min(node_colors), cmax=max(node_colors)))
		fig.show()


	def gen_message_pass_graph(self, rooms_file_dir:str, obj_file_dir:str):
		"""
		Generates a graph encoding all the relationships between rooms, objects, and furniture items.
		Their weights are a relation "score" that is intended to be interpreted later on as a probability.
		"""
		def get_classes_from_file(file_path):
			with open(file_path, 'r') as file:
				source = file.read()

			tree = ast.parse(source)
			class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

			return class_names

		room_data_classes = get_classes_from_file(rooms_file_dir)
		obj_data_classes = get_classes_from_file(obj_file_dir)
		# return room_data_classes, obj_data_classes

		faux_neural_graph = nx.Graph()

		for room in room_data_classes:
			if not faux_neural_graph.has_node("room"):
				faux_neural_graph.add_node(room)
			
			adj_rooms_dict = getattr(house_rooms, room).room_associations
			for adj_room, prob in adj_rooms_dict.items():
				if not faux_neural_graph.has_node(adj_room):
					faux_neural_graph.add_node(adj_room)	
				faux_neural_graph.add_edge(room, adj_room, weight=prob)
				
			assoc_furn_dict = getattr(house_rooms, room).furniture_associations
			for furn_item, prob in assoc_furn_dict.items():
				if not faux_neural_graph.has_node(furn_item):
					faux_neural_graph.add_node(furn_item)
				faux_neural_graph.add_edge(room, furn_item, weight=prob)

			assoc_obj_dict = getattr(house_rooms, room).object_associations
			for obj, prob in assoc_obj_dict.items():
				if not faux_neural_graph.has_node(obj):
					faux_neural_graph.add_node(obj)
				faux_neural_graph.add_edge(room, obj, weight=prob)
		
		for obj in obj_data_classes:
			obj_furn_dict = getattr(house_objects, obj).furniture_associations
			for furn_item, prob in obj_furn_dict.items():
				faux_neural_graph.add_edge(obj, furn_item, weight=prob)
		
		return faux_neural_graph