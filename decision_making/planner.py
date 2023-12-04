"""
Isaac Remy
Control and Trustworthy Robotics Laboratory
Last updated: 8/29/2023
iremy@uw.edu
"""

from . import util_classes
from . import util_functions
from .  import gen_models
from . import bayes_models
import random as rd
from copy import deepcopy
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt


RANDOM = 0
MAX_DECISIONS = 20
DISTANCE_SCALER = 1

class Planner:
	def __init__(self, params:util_classes.PlannerParams, gen_model:callable, bayes_update:bayes_models.BayesianUpdater, 
	      			termination_condition:callable, external_controller:callable=None, distance_func:callable=None,
					world:object=None, animation=None, true_graph = None, greedy_policy=False) -> None:
		"""
		Responsible for gathering observations and executing a subsequent path; 
		"""
		self.agent_state = params.agent_state
		self.env_graph = params.env_graph # Represents the observed env graph
		self.estimation_variable = params.estimation_variable
		self.root = params.root
		self.cost_scaler = params.cost_scaler
		self.horizon_len =  params.horizon_len
		self.trajectory_samples = params.trajectory_samples
		self.value_func = params.value_func
		self.observations = params.observation_set
		self.gen_model = gen_model
		self.bayes_update_class = bayes_update
		self.termination_condition = termination_condition
		self.updater = None
		self.prior = {}
		self.tried = set()
		# Our class's default decision execution function is designed for real-world testing
		# External controller is designed to be used for testing in more artificial enviroments where "continuity" is not important
		self.decision_exec = self.__controller if external_controller == None else external_controller
		self.distance_func = distance_func
		self.world = world
		self.animation = animation
		self.true_graph = true_graph
		self.greedy_polcy = greedy_policy

	
	def planner(self, event=None) -> None:
		"""
		High-level planner; initializes an infinite loop that continues until
		a termination criteria is specified.
		"""
		self.env_graph, self.prior = self.gen_model(self.env_graph, self.estimation_variable, self.root, self.tried)
		# print(self.env_graph.nodes())
		true_name = "_".join(re.findall("[a-zA-Z]+", self.estimation_variable))
		
		accumulated_travel_cost = 0.0
		num_decisions = 0
		
		decisions = []
		# orig_ent = util_functions.shannon_entropy(self.prior)
		
		while True:
			if self.animation is not None:
				print()
				print(self.prior)
				print()

			decision = ""
			steps_path = []

			# Greedy policy, i.e. pick the location with the immediate highest likelihood
			# Equates to a horizon length of one. This behavior would show up in the existing planner,
			# but computation is quicker this way
			if self.greedy_polcy:
				# self.updater = self.bayes_update_class(list(self.prior.keys()))
				decision = self.max_likelihood_baseline()[0]
				
				if RANDOM == 1:
					decision = rd.sample(list(self.prior.keys()), 1)[0]

				if self.distance_func is not None and self.world is not None:
					print("I chose the " + decision)
					print("Planning path...")
					_, steps_path = self.distance_func(self.agent_state.position, 
				      self.env_graph.nodes[decision]["centroid"], self.world)
			
			else:
				decision, _, _, steps_path = self.__compute_traj(num_decisions)
		
			
			decisions.append(decision)
			accumulated_travel_cost += util_functions.dijkstra_path(self.env_graph, self.agent_state.semantic_state,
														    decision, po=False)[2]

			if self.animation is not None:
				self.agent_state.position = steps_path[-1]
				self.animation.set_attrs(steps_path) #
			
			# print("+ + + + + + + + + + + + + + +")
			# Is only ever going to be true if terminating early is an option
			if decision == "null" or num_decisions == MAX_DECISIONS:
				print("FAILED")
				# container = [node for node in self.true_graph.neighbors(true_name + "0")][0]
				# print(f"obj: {true_name}, true container: {container}")
				# print(f"prior: {self.prior[container]}")
				# print(f"seq: {str(decisions)}")
				return 0, accumulated_travel_cost, num_decisions # 0 for failure/terminated early
			
			num_decisions += 1

			# Update the environment graph with new information we gather in making this decision and
			# get the observation.
			self.env_graph, self.agent_state.semantic_state, obs = self.decision_exec(decision, self.estimation_variable, self.env_graph)

			# Terminate if our observation meets the criteria to do so
			if self.termination_condition(obs):
				if self.animation is not None:
					print(f"I found the {self.estimation_variable} in {decision}!")
				# print(f"seq: {str(decisions)}")
				return 1, accumulated_travel_cost, num_decisions # 1 for success
			
			posterior = {}
			if decision in self.prior.keys():
				# posterior = self.updater.infer(self.prior, {decision: obs})
				self.tried.add(decision)
			else:
				posterior = self.prior
			# if self.animation is not None:
			# 	print("KL DIVERGENCE: " + str(util_functions.kl_divergence(posterior, self.prior, 1)))

			# TODO: add new leaves to prior upon observing a container (current model doesn't do this)
			self.env_graph, self.prior = self.gen_model(self.env_graph, self.estimation_variable, self.root, self.tried)
			
			if self.animation is not None:
				event.wait()

	
	def __compute_traj(self, t) -> list:
		"""
		Finds a single sequence of decisions to make by sampling many different decision trajectories
		and picking the one expected to minimize relative entropy measure and travel cost
		"""
		# Get an estimated graph up to the estimated variable's layer(s),
		# along with a prior over all decisions and their relevance to the estimation variable 

		# self.updater = self.bayes_update_class(list(self.prior.keys()))
		orig_prior = deepcopy(self.prior)

		use_world = self.distance_func is not None and self.world is not None

		best_decisions = []

		for j in range(10):
			total_costs = []
			traj_costs = []
			all_paths = []
			graph_copy = deepcopy(self.env_graph)

			graph_copy = gen_models.complete_partial_graph(graph_copy)
			graph_copy, prior = self.gen_model(graph_copy, self.estimation_variable, self.root, self.tried)
			graph_copy.remove_node(self.root)

			# Sample a bunch of decision-trajectories 
			for i in range(self.trajectory_samples):
				decision_seq = [self.agent_state.semantic_state]
				prior_keys = list(prior.keys())
				
				while len(decision_seq) < (self.horizon_len + 1):
					decision = rd.sample(prior_keys, 1)[0]

					while decision in self.tried:
						decision = rd.sample(prior_keys, 1)[0]

					decision_seq.append(decision)

				self.paths = {}
				tree_ev, _ = self.__compute_rollout(decision_seq, deepcopy(prior), 0, graph_copy)
				dijkstra_result = util_functions.dijkstra_path(graph_copy, decision_seq[0], decision_seq[1], po=True)
				
				steps_path = [] # use if planning in real world. list of grid-cells

				# traj_costs.append(dijkstra_result[2]) # TODO: distance scaler only for when testing
				total_costs.append(tree_ev) # what we minimize over
				
				if decision_seq[1] in orig_prior.keys():
					all_paths.append(decision_seq[1])
				else:
					idx = 2
					decision = dijkstra_result[1][1]
					while dijkstra_result[1][idx] in orig_prior.keys():
						decision = dijkstra_result[1][idx]
						idx += 1
					all_paths.append(decision)



			i = total_costs.index(max(total_costs))
			best = all_paths[i]# best decision 

			best_decisions.append(best)
		
		best = max(set(best_decisions), key=best_decisions.count)
		# print(f"best decisions: {best_decisions}")
		
		steps_path = None

		# Plan physical path (using something like a* planning)
		# Used if we are actually animating the behavior of the agent
		if use_world:
			print("I chose the " + best)
			print("Planning path...")
			_, steps_path = self.distance_func(self.agent_state.position, 
				      self.env_graph.nodes[best]["centroid"], self.world)

		return best, total_costs[i], traj_costs, steps_path
	

	def __compute_rollout(self, var_seq, prior:dict, current_depth, graph_copy):
		"""
		
		"""
		if current_depth == self.horizon_len - 1:
			return 0.0, 0.0
		
		travel_cost = 0.0
		expected_val = 0.0
		current_state = var_seq[current_depth]
		next_state = var_seq[current_depth+1]
		
		posterior = {key: val for key, val in prior.items() if key != current_state}
		if len(posterior) == 0 or current_state == next_state:
			return 0.0, 0.0
		
		norm = sum(posterior.values())
		posterior = {key: val / norm for key, val in posterior.items()}
		next_value = self.__compute_rollout(var_seq, posterior, current_depth + 1, graph_copy)
				
		p=prior[next_state]

		expected_val += p + next_value[0]

		travel_cost = util_functions.dijkstra_path(graph_copy, current_state, next_state, po=True)[0] * \
			self.cost_scaler[2] * (1 / DISTANCE_SCALER)
		
		expected_val -= travel_cost
		travel_cost += next_value[1]

		return expected_val, travel_cost


	# def open_loop_sampler(self, gam_1, gam_2):
	# 	all_seqs = []
	# 	values = []
	# 	for i in range(self.trajectory_samples):
	# 		decision_seq = []
	# 		value = 0.0
	# 		# decision_seq[0] = self.agent_state.semantic_state
	# 		for j in range(self.horizon_len):
	# 			decision = rd.choices([key for key in self.prior.keys()])[0]
	# 			while decision in decision_seq:
	# 				decision = rd.choices([key for key in self.prior.keys()])[0]
	# 				# continue
	# 			decision_seq.append(decision)
	# 			# value += ((gam_1 ** j) * self.prior[decision]) + ((1 - self.prior[decision]) / (gam_2 * self.prior[decision]))
	# 			value += self.prior[decision]

	# 		all_seqs.append(decision_seq)
	# 		values.append(value)

	# 	i = values.index(max(values))
	# 	return all_seqs[i]

	
	def max_likelihood_baseline(self):
		return sorted(self.prior, key=lambda x: self.prior[x], reverse=True)[:self.horizon_len + 1]
			


	def __controller(self):
		# TODO
		pass