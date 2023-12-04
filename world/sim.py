from decision_making.util_classes import AgentState
from . import build_houses
import networkx as nx
import re
import cv2
import pickle
import threading
from threading import Thread
import pygame
import numpy as np
from . import house_furniture as hf
from time import sleep
from copy import deepcopy

GRID_SIZE = 37
SIDEBAR_WIDTH = 100

class Sim():
    def __init__(self, real_env=None, real_grid=None) -> None:
        self.real_env = real_env
        self.real_grid = real_grid
        
        # Used when animated agent movement
        self.current_path = [] # the agent path
        self.message = "Idle" # the status-message we display during simulation
        self.path_lock = threading.Lock()


    def set_attrs(self, path):
        with self.path_lock:
            # print("acquired lock")
            # self.message = message
            self.current_path = deepcopy(path)


    def animate_agent(self, agent_start, planning_event):
        """
        Animates the agent moving along its path in real-time. Is meant to be threaded with the planner function.
        """
        # TODO: create separate function for PyGame initialization  
        # Initialize Pygame
        pygame.init()
        orig_grid = deepcopy(self.real_grid)
        # Grid and message area settings
        GRID_SIZE = 37
        num_rows = len(self.real_grid)
        num_cols = len(self.real_grid[0])
        message_area_height = 100

        # Get a list of furniture (since we want to add color to them)
        furniture = [node for node in self.real_env.nodes() if nx.get_node_attributes(self.real_env, "class")[node] == "furniture"]

        # Create a Pygame window
        # window_size = (num_cols * GRID_SIZE + SIDEBAR_WIDTH, num_rows * GRID_SIZE + message_area_height)
        window_size = ((num_cols * GRID_SIZE) + 1, (num_rows * GRID_SIZE) + 1)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Grid-World Environment")

        # Set up a font for displaying messages TODO: figure out how to reference font file
        font = pygame.font.Font("/home/isaac/isogrids/src/house_env/Roboto-Black.ttf", 30)

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BEIGE = (237, 202, 161)
        LINE_COLOR = (181, 193, 204)
        AGENT = (0, 220, 49)

        # Define options and their corresponding colors/values
        x = lambda tup: tuple(tup[::-1])

        options = [{"label": "floor", "color": BEIGE, "value": 0},
                   {"label": "wall", "color": BLACK, "value": 1},]
        
        value_to_furniture_dict = {idx: node for idx, node in enumerate(furniture, 2)}

        additional = [{"label": node, "color":  x(getattr(hf, node[:-1]).color), "value":idx} for idx, node in value_to_furniture_dict.items()]
        options.extend(additional) 
        options.append({"color":AGENT, "value": len(options)})

        last_pos = agent_start
        print(agent_start)
        self.real_grid[last_pos[0]][last_pos[1]] = options[-1]["value"]

        # Game loop
        running = True
        while running:
            path_copy = []

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Clear the screen
            screen.fill(WHITE)
            
            # Draw grid cells based on the grid_matrix
            for row in range(num_rows):
                for col in range(num_cols):
                    cell_color = options[self.real_grid[row, col]]["color"]
                    if self.real_grid[row, col] != len(options) - 1:
                        pygame.draw.rect(screen, cell_color, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                    else:
                        pygame.draw.rect(screen, options[orig_grid[row, col]]["color"], (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))
                        pygame.draw.rect(screen, cell_color, ((col * GRID_SIZE) + 5, (row * GRID_SIZE) + 5, GRID_SIZE - 10, GRID_SIZE - 10))
            
            self.__draw_grid_lines(screen, num_rows, num_cols, window_size, LINE_COLOR)

            planning_event.clear()

            for row, col in self.current_path:
                self.real_grid[last_pos] = orig_grid[last_pos]
                prev_color = options[orig_grid[last_pos]]["color"]
                pygame.draw.rect(screen, prev_color, ((last_pos[1] * GRID_SIZE) + 1, (last_pos[0] * GRID_SIZE) + 1, GRID_SIZE - 1, GRID_SIZE - 1))
                
                last_pos = (row, col)
                self.real_grid[last_pos] = len(options) - 1
                pygame.draw.rect(screen, AGENT, ((col * GRID_SIZE) + 5, (row * GRID_SIZE) + 5, GRID_SIZE - 10, GRID_SIZE - 10))
                pygame.display.update()
                sleep(0.4)  # Delay for animation effect TODO: make a constant
            
            planning_event.set()
            
            self.current_path = []

            # Display messages in the message area
            # message = self.message
            # text_surface = font.render(message, True, BLACK)
            # screen.blit(text_surface, (10, num_rows * GRID_SIZE + 25))

            # Update the display (TODO: use update instead? renders faster)
            pygame.display.flip()

        # Quit Pygame
        pygame.quit()



    def decision_outcome(self, decision, obj, graph):
        """
        Simulated 'observation' function; given a decision, reveals the result of the decision 
        based on the known underlying world-graph, along with the new agent state and the observation (1 or 0).
        The revealed nodes are attached to the returned graph.
        """
        agent_state = AgentState(decision, nx.get_node_attributes(graph, "centroid")[decision])
        neighbors = [node for node in self.real_env.neighbors(decision)]
        attrs = {neighbor: self.real_env.nodes[neighbor] for neighbor in neighbors}
        obs = 0

        # Parse to check if the target object has been observed,
        if obj in [" ".join(re.findall("[a-zA-Z]+", node)) for node in neighbors]:
            obs = 1

        edges = [(decision, neighbor) for neighbor in neighbors]
        graph.add_edges_from(edges)
        nx.set_node_attributes(graph, attrs)

        return graph, agent_state.semantic_state, obs


    def __construct_occupancy_grid(self, storage_dict):
        """
        Helper function that takes the world-map image of the environment and returns and occupancy
        grid reflecting whether or not a space is occupied.
        """
        image = storage_dict["world"]

        # Resize the image to match the grid size
        height, width, _ = image.shape
        new_height = (height // GRID_SIZE) * GRID_SIZE
        new_width = (width // GRID_SIZE) * GRID_SIZE
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert the resized image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Create an empty canvas for the discretized color image
        discretized_color_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Iterate over the grid squares and calculate average intensity
        for y in range(0, new_height, GRID_SIZE):
            for x in range(0, new_width, GRID_SIZE):
                grid = gray_image[y:y+GRID_SIZE, x:x+GRID_SIZE]
                average_intensity = np.mean(grid)
                
                if average_intensity > 128:  # Threshold to determine black or white
                    discretized_color_image[y:y+GRID_SIZE, x:x+GRID_SIZE] = [255, 255, 255]

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(discretized_color_image, cv2.COLOR_BGR2GRAY)

        # Calculate the number of rows and columns in the grid
        rows, cols = gray_image.shape

        # Calculate the number of grid cells in rows and columns
        num_rows = rows // GRID_SIZE
        num_cols = cols // GRID_SIZE

        # Create an empty binary matrix
        binary_matrix = np.ones((num_rows, num_cols), dtype=np.uint8)

        # Iterate over the grid cells and determine whether they are light or dark
        for row in range(num_rows):
            for col in range(num_cols):
                grid_cell = gray_image[row * GRID_SIZE : (row + 1) * GRID_SIZE, col * GRID_SIZE : (col + 1) * GRID_SIZE]
                if np.any(grid_cell > 0):  # Check if any pixel in the cell is not black
                    binary_matrix[row, col] = 0

        return binary_matrix, image


    def modify_graph(self, file_path:str):
        """
        Modifies the node attributes of a scene-graph (loaded from a pickle file, along with its original meta-data) to
        fit the dimensions of a grid-world version of its environment
        :param file_path: the path of the pickle file containing the scene-graph, and its associated meta-data.
        """
        # Load the map + scene-graph
        storage_dict = pickle.load(open(file_path, 'rb'))
        meta_data = storage_dict["data"]
        scene_graph =  nx.Graph(storage_dict["scene_graph"])
       
        # Add node-attributes to all the nodes in the scene-graph so
        # we can access them through the planner easily
        attr_dict = {}
        for node in meta_data:
            connection_surfaces = []
            if meta_data[node]["class"] == "room":
                connection_surfaces = meta_data[node]["doorways"]

            attr_dict[meta_data[node]["name"]] = {"class": meta_data[node]["class"], "centroid": meta_data[node]["centroid"], 
                                                "contour": meta_data[node]["contour"], "connection_surfaces": connection_surfaces}
        
        # print(scene_graph.nodes())
        nx.set_node_attributes(scene_graph, attr_dict)
        
        # x=["vacuum_cleaner0", "vacuum_cleaner1", "vacuum_cleaner2"]
        # scene_graph.remove_nodes_from(x)
        # print(scene_graph.nodes())
        # h=build_houses.HouseBuilder()
        # meta_data_copy = deepcopy(meta_data)
        # for node in meta_data.keys():
        #     if meta_data_copy[node]["name"] in x:
        #         del meta_data_copy[node]
        # h.visualize_graph(meta_data_copy, scene_graph)

        # print(scene_graph.nodes["living_room0"]["centroid"])
        # print(scene_graph.nodes["living room0"]["centroid"])
        # print(scene_graph.nodes)

        # Get the occupancy grid of the world we load
        m, img = self.__construct_occupancy_grid(storage_dict)

        def cv_show(img):
            """
            Displays img. Designed for threading.
            """
            cv2.imshow("OpenCV Window", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Threading sub-class (used for displaying original world and edit GUI at the same time) that allows
        # us to access a returned value from a threaded function
        class ThreadWithReturnValue(Thread):
            def __init__(self, group=None, target=None, name=None,
                        args=(), kwargs={}, Verbose=None):
                Thread.__init__(self, group, target, name, args, kwargs)
                self._return = None

            def run(self):
                if self._target is not None:
                    self._return = self._target(*self._args,
                                                        **self._kwargs)
            def join(self, *args):
                Thread.join(self, *args)
                return self._return

        # Multi-threading to display modification GUI and reference world-map at the same time.
        opencv_thread = threading.Thread(target=cv_show, args=(img,))
        pygame_thread = ThreadWithReturnValue(target=self.__modify_env, args=(m, scene_graph))

        opencv_thread.start()
        pygame_thread.start()

        opencv_thread.join()
        m, d = pygame_thread.join()

        # Internal function to extract the integer locations of the room/furniture centroids
        # from a grid.
        def convert_centroids_from_grid(m, d, attr_dict:dict):
            centroids = set()
            num_rows = len(m)
            num_cols = len(m[0])
            for row in range(num_rows):
                for col in range(num_cols):
                    if m[row][col] != 0 and m[row][col] != 1 and \
                    d[m[row][col]] not in centroids:
                        centroids.add(d[m[row][col]])
                        attr_dict[d[m[row][col]]]["centroid"] = (row, col) # y, x
            
            rooms = [node for node in attr_dict.keys() if attr_dict[node]["class"] == "room"]

            for room in rooms:
                attr_dict[room]["centroid"] = tuple([int(element / GRID_SIZE) 
                                                    for element in attr_dict[room]["centroid"][::-1]]) # iterate through reversed tuple because 
                                                                                                        # original centroid value is (x, y)            
        # Call internal helper function
        convert_centroids_from_grid(m, d, attr_dict)
        
        # Save the scene-graph along with its modified meta-data
        storage_dict = {"grid world": m, "node attrs": attr_dict, "scene graph": nx.to_dict_of_dicts(scene_graph)}
        file_dir = './src/house_env/final_houses'
        pickle.dump(storage_dict, open(file_dir + f'/test.pickle', 'wb'))

    
    def __modify_env(self, grid_matrix, G:nx.Graph):
        """
        Internal function that creates the interaction tool allowing us to modify the cells of a world
        """
        # Initialize Pygame
        pygame.init()

        # Grid and message area settings
        GRID_SIZE = 37
        num_rows = len(grid_matrix)
        num_cols = len(grid_matrix[0])
        message_area_height = 100

        # Colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        BEIGE = (237, 202, 161)
        GRAY = (200, 200, 200)

        furniture = [node for node in G.nodes() if nx.get_node_attributes(G, "class")[node] == "furniture"]

        # Create a Pygame window
        window_size = (num_cols * GRID_SIZE + SIDEBAR_WIDTH, num_rows * GRID_SIZE + message_area_height)
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("IsoGrid-World Editor")

        # Set up a font for displaying messages TODO: figure out how to reference font file
        font = pygame.font.Font("/home/isaac/isogrids/src/house_env/Roboto-Black.ttf", 30)

        # Define options and their corresponding colors/values

        x = lambda tup: tuple(tup[::-1])

        options = [{"label": "floor", "color": BEIGE, "value": 0},
                   {"label": "wall", "color": BLACK, "value": 1}]
        
        value_to_furniture_dict = {idx: node for idx, node in enumerate(furniture, 2)}

        additional = [{"label": node, "color":  x(getattr(hf, node[:-1]).color), "value":idx} for idx, node in value_to_furniture_dict.items()]
        options.extend(additional)

        selected_option_index = 0  # Default option selected

        # Game loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.pos[0] > (GRID_SIZE * num_cols):  # Click in the sidebar
                                selected_option_index = event.pos[1] // GRID_SIZE
                    else:  # Click in the grid
                        row = event.pos[1] // GRID_SIZE
                        col = (event.pos[0]) // GRID_SIZE
                        grid_matrix[row, col] = options[selected_option_index]["value"]

            # Clear the screen
            screen.fill(WHITE)

            # draw sidebar
            self.__draw_sidebar(options, font, screen, num_cols)

            # Draw grid cells based on the grid_matrix
            for row in range(num_rows):
                for col in range(num_cols):
                    cell_color = options[grid_matrix[row, col]]["color"]
                    pygame.draw.rect(screen, cell_color, (col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE))

            self.__draw_grid_lines(screen, num_rows, num_cols, window_size, (181, 193, 204))

            # Display messages in the message area
            message = "Selected: " + options[selected_option_index]["label"]
            text_surface = font.render(message, True, BLACK)
            screen.blit(text_surface, (10, num_rows * GRID_SIZE + 25))

            # Update the display
            pygame.display.flip()

        # Quit Pygame
        pygame.quit()

        # print(value_to_furniture_dict)
        return grid_matrix, value_to_furniture_dict


    def __draw_sidebar(self, options, font, screen, num_cols):
        for i, option in enumerate(options):
            button_color = option["color"]
            label = option["label"]
            text_surface = font.render(label, True, button_color)
            button_rect = pygame.Rect(num_cols * GRID_SIZE, i * GRID_SIZE, SIDEBAR_WIDTH, GRID_SIZE)
            pygame.draw.rect(screen, button_color, button_rect)
            screen.blit(text_surface, (10, i * GRID_SIZE))


    # Function to draw grid lines
    def __draw_grid_lines(self, screen, num_rows, num_cols, window_size, color):
        for row in range(num_rows + 1):
            pygame.draw.line(screen, color, (0, row * GRID_SIZE), (num_cols * GRID_SIZE, row * GRID_SIZE))
        for col in range(num_cols + 1):
            pygame.draw.line(screen, color, (col * GRID_SIZE, 0), (col * GRID_SIZE, num_rows * GRID_SIZE))