from dataclasses import dataclass
import networkx as nx
import re

@dataclass
class AgentState:
    """
    An agent state is broken up into two variables.\n
    Args:
        semantic_state (str): Describes the high-level location of the agent, eg. "living_room".
        position (tuple): The position of the agent in Cartesian coordinates.
    """
    semantic_state: str
    position: tuple


@dataclass
class PlannerParams:
    """
    Stores necessary initialization parameters for the planner.\n
    Args:
        agent_state (AgentState): The state of the agent at initialization.
        env_graph (nx.Graph): The graphical model of the environment at initialization (eg. 3D scene-graph).
        estimation_variable (str): The thing whose high-level state(s) we wish to estimate.
        root (str): The label for the root of the environment graph/tree (eg. "house")
        reward_scaler (float): An user-defined parameter for how we should scale our reward function 
        cost_scaler (float): An user-defined parameter for how we should scale our cost (intended for travel) 
        observation_set (set): Set of possible observations for any decision-state.
        horizon_len (int): How long a sequence of planned decisions is.
        trajectory_samples (int): How many different decision trajectories we want to try.
    """
    agent_state: AgentState
    env_graph: nx.Graph
    estimation_variable: str
    root: str
    cost_scaler: float
    horizon_len: int
    trajectory_samples: int
    value_func: callable
    observation_set: set    

# @dataclass
class DecisionNode():
    """
    Stores the attributes of nodes for the decision-tree that's extracted from the 3D scene-graph.
    during planning.\n
    Args:
        name (str): Name of the node
        parent (str): The name of the node's parent
        children (tuple): Tuple of the node's children
        neighbors (tuple): Tuple of the node's neighbors
    """
    def __init__(self, name) -> None:
        self.name = name
        self.parent = "null"
        self.children = tuple()
        self.neighbors = tuple()


    def set_parent(self, parent) -> None:
        self.parent = parent


    def add_children(self, child) -> None:
        self.children += (child,)


    def add_neighbor(self, neighbor) -> None:
        self.neighbors += (neighbor,)

    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, DecisionNode):
            return (
                self.name == __value.name
                and self.parent == __value.parent
                and self.children == __value.children
                and self.neighbors == __value.neighbors
            )
        return False
    

    def __hash__(self) -> int:
        return hash((self.name, self.parent, self.children, self.neighbors))