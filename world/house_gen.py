from . import house_rooms, house_furniture, house_objects
# import house_rooms, house_furniture, house_objects
import networkx as nx
import random as rd
import numpy as np
import math
import cv2
from copy import deepcopy
import time
from .house_gen_timeout import timeout
# from house_gen_timeout import timeout

SCALE_X = 700
SCALE_Y = 700
NUM_REGIONS = 6
BLUR_KERNEL = 37
DOORWAY_WIDTH = 70
TIMEOUT = 15
OBJ_SPREAD_RADIUS = 43

class House:
	def __init__(self) -> None:
		self.__graph = nx.Graph()

	@timeout(TIMEOUT)
	def gen(self):
		return self.__gen_graph()
	

	def __gen_graph(self):

		def find_distance(p1, p2):
			"""
			Returns L2 distance between two points (x, y)
			"""
			return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

		inverted_image, contours, centroids = self.__gen_grid()

		height, width = inverted_image.shape
		color_image = np.zeros((height, width, 3), dtype=np.uint8)

		color_image[:, :, 0] = inverted_image  # Blue channel
		color_image[:, :, 1] = inverted_image  # Green channel
		color_image[:, :, 2] = inverted_image  # Red channel

		init_rooms_dict = {}
		for i in range(1, len(contours) + 1):
			init_rooms_dict[i] = {"contour": contours[i - 1], "centroid": centroids[i - 1]} # last index represents shared-edges
		
		def is_point_near_edge(x, y, distance_threshold):
			"""
			Calculate distances from the point to a specified border (for edge of matrix) and return True if so, False otherwise.
			"""
			distance_left = x
			distance_right = SCALE_X - x - 1
			distance_top = y
			distance_bottom = SCALE_Y - y - 1
			
			# Check if the point is within the distance threshold from any edge
			if (distance_left <= distance_threshold or
				distance_right <= distance_threshold or
				distance_top <= distance_threshold or
				distance_bottom <= distance_threshold):
				return True
			else:
				return False
				
		def find_midpoint(point1, point2):
			x_mid = (point1[0] + point2[0]) / 2
			y_mid = (point1[1] + point2[1]) / 2
			return (x_mid, y_mid)

		def find_closest_points(points):
			closest_points = {}
			for i in range(len(points)):
				current_point = tuple(points[i])
				min_distance = float('inf')
				closest_point = None

				for j in range(len(points)):
					if i != j:
						dist = find_distance(current_point, points[j])
						if dist < min_distance:
							min_distance = dist
							closest_point = tuple(points[j])

				closest_points[current_point] = closest_point

			return closest_points
		
		midpoints = []
		connected_rooms = set()
		midpoints_to_rooms  = {}

		for room in init_rooms_dict.keys():
			connected_point_sets = []

			# Iterate through the points in the contour
			for i in range(1, len(init_rooms_dict[room]["contour"])):
				
				point = init_rooms_dict[room]["contour"][i][0]
				prev_point = init_rooms_dict[room]["contour"][i - 1][0]

				# Calculate the distance between the current point and the previous point
				dist = find_distance(point, prev_point)

				both_edge = False
				if is_point_near_edge(point[0], point[1], 50) and is_point_near_edge(prev_point[0], prev_point[1], 50):
					both_edge = True

				# Check if the distance is greater than the threshold
				if dist > (DOORWAY_WIDTH * 1.9) and not both_edge: #:
					# If the distance is greater than the threshold, start a new segment
					connected_point_sets.append([point, prev_point])

			point = init_rooms_dict[room]["contour"][-1][0]
			prev_point = init_rooms_dict[room]["contour"][0][0]
			dist = find_distance(point, prev_point)
			both_edge = False
			
			if is_point_near_edge(point[0], point[1], 50) and is_point_near_edge(prev_point[0], prev_point[1], 50):
				both_edge = True
			if dist > (DOORWAY_WIDTH * 2) and not both_edge:
				connected_point_sets.append([point, prev_point])

			for p_set in connected_point_sets:
				midpoint = np.int_(find_midpoint(p_set[0], p_set[1]))
				midpoints.append(midpoint)
				midpoints_to_rooms[tuple(midpoint)] = room
		
		closest_points = find_closest_points(midpoints)

		rooms_to_doorways = {}

		# Create list of edges based on the closest points between all the edge-midpoints.
		# These will inform the location of doorways between rooms.
		for point, closest_point in closest_points.items():
			if closest_points[closest_point] == point:
				doorway = np.int_(find_midpoint(point, closest_point))
				closest_room = midpoints_to_rooms[closest_point]
				this_room = midpoints_to_rooms[point]
				if all(color_image[doorway[1], doorway[0]] == (0, 0, 0)) and (closest_room, this_room) not in connected_rooms:
					# doorways.append(doorway)
					for room_p in (closest_room, this_room):
						if room_p not in rooms_to_doorways.keys():
							rooms_to_doorways[room_p] = (doorway,)
						else:
							rooms_to_doorways[room_p] += (doorway,)
					
					connected_rooms.add((midpoints_to_rooms[point], midpoints_to_rooms[closest_point]))
					cv2.rectangle(color_image, (int(doorway[0] - BLUR_KERNEL / 1.5), int(doorway[1] + BLUR_KERNEL / 1.5)), (int(doorway[0] + BLUR_KERNEL / 1.5), int(doorway[1] - BLUR_KERNEL / 1.5)), (255, 255, 255), thickness=-1)

		graph_obj_dict = {}

		# Construct updated dictionary containing only valid rooms, i.e. those with edges
		for connection in connected_rooms:
			room1, room2 = connection
			for room in [room1, room2]:
				if room not in graph_obj_dict.keys():
					attributes = init_rooms_dict[room]
					doorways = rooms_to_doorways[room]
					edges = (room2,) if room == room1 else (room1,)
					attributes["edges"] = edges
					attributes["doorways"] = doorways
					attributes["class"] = "room"
					graph_obj_dict[room] = attributes
				else:
					graph_obj_dict[room]["edges"] += (room2,) if room == room1 else (room1,)

		print("Num Doorways: ", str(len(connected_rooms)))
		print("Num Rooms: ", str(len(graph_obj_dict)))

		max_room_id, _ = max(graph_obj_dict.items(), key=lambda item: cv2.contourArea(item[1]["contour"]))

		graph_obj_dict[max_room_id]["name"] = "living_room0"

		def dfs(room_dict, start_node):
			"""
			Populates the room layer of the scene-graph procedurally, given a dictionary of edges between the rooms
			and the starting node to generate from (living room is probably best).
			"""
			visited = set()
			queue = [start_node]
			visited.add(start_node)
			occurances_dict = {room_dict[start_node]["name"]: 1}
			self.__graph.add_edge("house", room_dict[start_node]["name"])

			while queue:
				node = queue.pop(0)
				neighbors = room_dict[node]["edges"]
				adj_rooms, adj_room_scores = zip(*getattr(house_rooms, room_dict[node]["name"][:-1]).room_associations.items())

				for neighbor in neighbors:
					if neighbor not in visited:
						room_name = rd.choices(adj_rooms, adj_room_scores)[0] + "0"

						duplicate = getattr(house_rooms, room_name[:-1]).duplicate
						again = 0
						while(self.__graph.has_edge("house", room_name) and again == 0):
							duplicate = getattr(house_rooms, room_name[:-1]).duplicate
							if duplicate > 0.01:
								again = np.random.binomial(1, duplicate ** occurances_dict[room_name[:-1]])
								if again == 1:
									room_name = room_name[:-1] + str(occurances_dict[room_name[:-1]])
							else:
								room_name = rd.choices(adj_rooms, adj_room_scores)[0]
								room_name += "0"

						self.__graph.add_edge("house", room_name)
						self.__graph.add_edge(room_dict[node]["name"], room_name)
						room_dict[neighbor]["name"] = room_name
						
						if room_name[:-1] not in  occurances_dict.keys():
							occurances_dict[room_name[:-1]] = 1
						else:
							occurances_dict[room_name[:-1]] += 1

						visited.add(neighbor)
						queue.append(neighbor)
					else:
						self.__graph.add_edge(room_dict[node]["name"], room_dict[neighbor]["name"])

		dfs(graph_obj_dict, max_room_id)

		return_img = deepcopy(color_image)

		graph_obj_dict[0] = {"name": "house", "centroid": (int(SCALE_X / 2), int(SCALE_Y / 2)), "contour":(), "class":"building"}

		house_edges = tuple([i for i in range(1, len(graph_obj_dict) + 1)])

		graph_obj_dict[0]["edges"] = house_edges

		def rand_contour_point(contour, max_distance_from_edge, min_distance_from_edge, doorways, f_pieces):
			while True:
				# Get the bounding rectangle of the contour
				x, y, w, h = cv2.boundingRect(contour)
				
				# Generate a random point within the bounding rectangle
				random_point = (x + np.random.randint(0, w), y + np.random.randint(0, h))
				
				# Check if the random point is within the contour and has the specified distance from its edges
				p_dist = cv2.pointPolygonTest(contour, random_point, True)
				if p_dist < max_distance_from_edge and p_dist > min_distance_from_edge:
					near_doorway = False
					for doorway in doorways:
						if find_distance(random_point, doorway) < (DOORWAY_WIDTH * 2):
							near_doorway = True
							break
					
					near_furniture = False
					for f in f_pieces:
						if find_distance(random_point, f) < (DOORWAY_WIDTH):
							near_furniture = True
							break

					if not near_doorway and not near_furniture:
						return random_point

		max_id = max(graph_obj_dict.keys())
		rooms = list(graph_obj_dict.keys())
		furniture_nums = {}

		def fully_connect_children(graph, children_nodes):
			# graph.add_node(parent_node)

			for child_node in children_nodes:
					for other_child in children_nodes:
							if child_node != other_child:
									graph.add_edge(child_node, other_child)

		for room_id in rooms:
			if room_id != 0:
				f_pieces = tuple()
				room_f_dict = getattr(house_rooms, graph_obj_dict[room_id]["name"][:-1]).furniture_associations.keys()
				num_pieces = len(room_f_dict)
				for f_id, f in enumerate(room_f_dict, max_id + 1):
					prob = getattr(house_rooms, graph_obj_dict[room_id]["name"][:-1]).furniture_associations[f]
					placed = np.random.binomial(1, prob)
					if placed == 1:
						if f not in furniture_nums.keys():
							furniture_nums[f] = 0
						else:
							furniture_nums[f] += 1
						f_name = f + str(furniture_nums[f])
						contour = graph_obj_dict[room_id]["contour"]
						point = rand_contour_point(contour, 30, 5, graph_obj_dict[room_id]["doorways"], f_pieces)
						f_pieces += (point,)
						graph_obj_dict[f_id] = {"name":f_name, "centroid":point, "edges": (room_id), "contour":(), "class":"furniture"}
						graph_obj_dict[room_id]["edges"] += (f_id,)
						self.__graph.add_edge(graph_obj_dict[room_id]["name"], f_name)
						cv2.circle(color_image, point, 5, (255, 0, 0), -1)
				
				# Fully connect the furniture for each room
				furniture_children = []
				neighbors = graph_obj_dict[room_id]["edges"]
				for node in neighbors:
					if graph_obj_dict[node]["class"] == "furniture":
						furniture_children.append(graph_obj_dict[node]["name"])
				
				fully_connect_children(self.__graph, furniture_children)

				max_id += num_pieces

		def random_point_on_circle(center_x, center_y, r):
				# Generate a random angle in radians
				angle = rd.uniform(0, 2*math.pi)

				# Calculate the x and y coordinates using polar-to-Cartesian conversion
				x = center_x + r * math.cos(angle)
				y = center_y + r * math.sin(angle)

				return x, y

		max_id = max(graph_obj_dict.keys()) + 1
		object_nums = {}

		# Place objects in scene-graph
		for item in rooms:
			if graph_obj_dict[item]["class"] == "room":
				possible_objects = getattr(house_rooms, graph_obj_dict[item]["name"][:-1]).object_associations
				max_object_types = len(possible_objects)
				num_objs = rd.randint(3, 10)
				obj_list = list(possible_objects.keys())
				for i in range(num_objs):
					obj = rd.choice(obj_list)
					room_obj_prob = possible_objects[obj]
					furniture_probs = getattr(house_objects, obj).furniture_associations
					if len(furniture_probs) > 0:
						neighbors = graph_obj_dict[item]["edges"]
						for node in neighbors:
							if graph_obj_dict[node]["class"] == "furniture":
								frn_name = graph_obj_dict[node]["name"][:-1]
								if frn_name in furniture_probs:
									obj_prob = room_obj_prob * furniture_probs[frn_name]
									if obj in object_nums:
										obj_prob *= (getattr(house_objects, obj).duplicate ** (object_nums[obj] + 1)) # Geometric decrease via duplicate value
									placed = np.random.binomial(1, obj_prob)
									if placed == 1:
										if obj not in object_nums:
											object_nums[obj] = 0
										else:
											object_nums[obj] += 1
										obj_name = obj + str(object_nums[obj])
										x_y = graph_obj_dict[node]["centroid"]
										obj_centroid = random_point_on_circle(x_y[0], x_y[1], 35)
										graph_obj_dict[max_id] = {"name":obj_name, "centroid":obj_centroid, "edges": (node), "contour":(), "class":"object"}
										self.__graph.add_edge(graph_obj_dict[max_id]["name"], graph_obj_dict[node]["name"])
										max_id += 1
					# Case for when object won't located within/on a furniture item
					else:
						obj_prob = room_obj_prob
						placed = np.random.binomial(1, obj_prob)
						if placed == 1:
							if obj not in object_nums:
								object_nums[obj] = 0
							else:
								object_nums[obj] += 1
							obj_name = obj + str(object_nums[obj])
							x_y = graph_obj_dict[item]["centroid"]
							obj_centroid = random_point_on_circle(x_y[0], x_y[1], OBJ_SPREAD_RADIUS)
							graph_obj_dict[max_id] = {"name":obj_name, "centroid":obj_centroid, "edges": (item), "contour":(), "class":"object"}
							self.__graph.add_edge(graph_obj_dict[max_id]["name"], graph_obj_dict[item]["name"])
							max_id += 1

		# Display locations of objects in map
		for obj in graph_obj_dict:
			if graph_obj_dict[obj]["class"] != "object":
				text = graph_obj_dict[obj]["name"]
				font = cv2.FONT_HERSHEY_SIMPLEX
				font_scale = 0.5
				font_color = (0, 0, 0)  # black color in BGR format
				line_thickness = 2
				text_size, _ = cv2.getTextSize(text, font, font_scale, line_thickness)
				position = graph_obj_dict[obj]["centroid"]
				text_x = position[0] - text_size[0] // 2
				text_y = position[1] + text_size[1] // 2
				cv2.putText(color_image, text, (text_x, text_y), font, font_scale, font_color, line_thickness)

		return return_img, color_image, graph_obj_dict, self.__graph


	def __gen_grid(self):
		'''
		Generates a randomly generated grid, where the integers filling the grid represent different types
		of rooms.
		'''
		floor_matrix = np.zeros((SCALE_Y, SCALE_X))

		def calculate_distance(row1, col1, row2, col2):
				return math.sqrt((row1 - row2)**2 + (col1 - col2)**2)

		def place_integers(matrix, num_integers, min_distance, max_distance):
				# Randomly place the first integer

			new_pos = {}
			row = rd.randint(0, len(matrix) - 1)
			col = rd.randint(0, len(matrix[0]) - 1)
			matrix[row][col] = 1

			new_pos[1] = (row, col)

			# Place the remaining integers
			for i in range(2, num_integers + 1):
				placed = False

				while not placed:
					# Randomly select a cell
					row = rd.randint(0, len(matrix) - 1)
					col = rd.randint(0, len(matrix[0]) - 1)

					# Check if the cell satisfies the distance constraint with existing integers
					valid_placement = True
					for r in range(len(matrix)):
						for c in range(len(matrix[0])):
							if matrix[r][c] != 0:
								distance = calculate_distance(row, col, r, c)
								if distance < min_distance or distance > max_distance:
									valid_placement = False
									break
						
						if not valid_placement:
							break

					if valid_placement:
						matrix[row][col] = i
						new_pos[i] = (row, col)
						placed = True

			return matrix, new_pos


		def generate_square_coordinates(distance):
			coordinates = []

			top_left = (-distance, distance)
			top_right = (distance, distance)
			bottom_left = (-distance, -distance)
			bottom_right = (distance, -distance)

			# Generate coordinates along each side of the square
			for x in range(top_left[0], top_right[0] + 1):
				coordinates.append((x, top_left[1]))

			for y in range(top_right[1], bottom_right[1] - 1, -1):
				coordinates.append((top_right[0], y))

			for x in range(bottom_right[0], bottom_left[0] - 1, -1):
				coordinates.append((x, bottom_right[1]))

			for y in range(bottom_left[1], top_left[1] + 1):
				coordinates.append((bottom_left[0], y))

			return coordinates


		floor_matrix, new_pos = place_integers(floor_matrix, NUM_REGIONS, int(0.25*SCALE_X), int(0.75*SCALE_X))
		num_zeros = (SCALE_X * SCALE_Y) - len(new_pos)
		distance = 1

		while num_zeros > 0:
			for idx, room in enumerate(new_pos):
				directions = generate_square_coordinates(distance)
				for dy, dx in directions:
					new_row = new_pos[room][0] + dy
					new_col = new_pos[room][1] + dx
					
					if 0 <= new_row < SCALE_Y and 0 <= new_col < SCALE_X:
						if floor_matrix[new_row][new_col] == 0:
							floor_matrix[new_row][new_col] = idx + 1
							num_zeros -= 1
			distance += 1

		def separate_polygons(matrix):
			integers = np.unique(floor_matrix)
			img_list = []
			cpy = deepcopy(matrix)
			for i in integers:
				cpy[cpy != i] = 0
				img_list.append(cpy)
				cpy=deepcopy(matrix)

			return img_list

		img_list = separate_polygons(floor_matrix)		

		def convert_to_image(grid, color_mapping):
				num_rows = len(grid)
				num_cols = len(grid[0])

				# Create an empty image array
				image = np.zeros((num_rows, num_cols, 3), dtype=np.uint8)

				# Iterate over the grid and assign colors based on the integer values
				for row in range(num_rows):
						for col in range(num_cols):
								integer = grid[row][col]
								color = color_mapping.get(integer, (0, 0, 0))  # Default to black if integer not in color mapping
								image[row, col] = color

				return image

		# Define a color mapping for integers
		color_mapping = {
				1: (0, 0, 255),   # Blue
				2: (0, 255, 0),   # Green
				3: (255, 0, 0),   # Red
				4: (0, 255, 255),  # Cyan
				5: (120, 120, 120),
				6: (65, 180, 15),
				7: (92, 23, 200),
				8: (200, 54, 10)
		}

		def convert_to_rectangle(image):
				# Convert the image to grayscale
				gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				
				# # Find contours of polygons in the grayscale image
				contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				
				# Create a new image to draw the rectangles
				result = np.zeros_like(gray)
				
				for contour in contours:
						# Find the bounding rectangle for each contour
						x, y, width, height = cv2.boundingRect(contour)
						
						# Fill the rectangle with the mean value of the polygon region
						result[y:y+height, x:x+width] = np.mean(gray[contour[:, :, 1], contour[:, :, 0]])
				
				return result
		
		start = time.time()
		size_img_dict = {}
		rects_list = []
		
		for idx, i in enumerate(img_list):
				max_size = 0
				cpy = convert_to_image(i, color_mapping)
				val = 0
				for x in range(len(i)):
					for y in range(len(i[0])):
						if i[x][y] != 0:
							val = i[x][y]
							break
				
				rect = convert_to_rectangle(cpy)
				black_image = np.zeros((len(i), len(i[0]), 3), dtype=np.uint8)
				for j in range(len(i)):
					for k in range(len(i[0])):
						if rect[j][k] != 0:
							black_image[j][k] = color_mapping[val]
							max_size += 1

				size_img_dict[idx] = max_size
				rects_list.append(black_image)
		
		end = time.time()
		print("Generation time: ", str(end-start))

		sorted_keys = sorted(size_img_dict, key=lambda k: size_img_dict[k], reverse=True)

		black_image = np.zeros((len(img_list[0]), len(img_list[0][0]), 3), dtype=np.uint8)
		for idx in sorted_keys:
			curr_img = rects_list[idx]
			for i in range(len(curr_img)):
				for j in range(len(curr_img[0])):
					if curr_img[i][j].any():
						black_image[i][j] = curr_img[i][j]

		gray = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)

		# Apply Canny edge detection to identify edges
		edges = cv2.Canny(gray, 100, 200)

		# Apply Gaussian blur to smooth the edges (which will represent the walls between rooms)
		blurred_edges = cv2.GaussianBlur(edges, (BLUR_KERNEL, BLUR_KERNEL), 0)  # You can adjust the kernel size for stronger or weaker smoothing

		# Set white pixels over the edges
		for i in range(len(blurred_edges)):
				for j in range(len(blurred_edges[0])):
						if blurred_edges[i, j] > 0:  # Check if the pixel value is greater than 0 (indicating an edge)
								blurred_edges[i, j] = 255  # Set the pixel to white
					
		border_size = (50, 50)


		# Add a white border with the same size as the Gaussian blur kernel
		bordered_image = cv2.copyMakeBorder(blurred_edges, border_size[0], border_size[0], border_size[1], border_size[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

		inverted_image = cv2.bitwise_not(bordered_image)

		contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Initialize a list to store the centroids of each polygon
		centroids = []

		# Loop through each identified contour (polygon)
		for contour in contours:
				# Calculate the moments of the contour
				M = cv2.moments(contour)

				# Calculate the x and y coordinates of the centroid
				centroid_x = int(M["m10"] / M["m00"])
				centroid_y = int(M["m01"] / M["m00"])

				# Append the centroid coordinates to the list
				centroids.append((centroid_x, centroid_y))

		return inverted_image, contours, centroids