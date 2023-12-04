from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.inference import VariableElimination
# import util_functions

# TODO
class BayesianUpdater:
	def __init__(self, variables:list, measurement_error:float=0.001) -> None:
		pass


class ObjBayesNet(BayesianUpdater):
	def __init__(self, variables:list, measurement_error:float=0.001) -> None:
		"""
		This example class is meant for object-search scenarios, where the state to be estimated is the container
		of a target object. 
		"""
		self.obs_cpds = [] # list of Bernoulli distributions for each leaf
		var_idx = 0
		vars_len = len(variables)
		self.struc = []
		
		for var in variables:
			self.struc.append(('state', var)) 
			neg_values = [measurement_error if i == var_idx else 1-measurement_error for i in range(vars_len)]
			pos_values = [1 - val for val in neg_values]
			# Construct the Bernoulli distribution for the variable, where:
			#	- there are two observation values (true and false)
			# 	- the true and false values (1 or zero +- measurement error)
			# 	- the evidence variable: leaf-state (true or false)
			# 	- number of evidence variables -- prior size
			curr_cpd = TabularCPD(variable=var, variable_card=2, values=[neg_values, pos_values],
			 						evidence=['state'], evidence_card=[vars_len])
			self.obs_cpds.append(curr_cpd)
			var_idx += 1


	def infer(self, prior, observation) -> dict:
		"""
		Given a prior-distribution and an observation on that prior, return the posterior distribtion.
		"""
		prior_len = len(prior)
		state_cpd = TabularCPD(variable='state', variable_card=prior_len, values=[[prior[var]] for var in prior.keys()])
		model = BayesianNetwork([('state', var) for var in prior.keys()])
		temp = [cpd for cpd in self.obs_cpds]
		temp.append(state_cpd)
		model.add_cpds(*temp)
		
		infer = VariableElimination(model)
		new_cpd = infer.query(variables=['state'], evidence=observation)
		# Create the posterior dictionary from the new cpd object
		posterior = {}
		variables = [variable.variables[0] for variable in self.obs_cpds]
		values = new_cpd.values.flatten()
		for i, variable in enumerate(variables):
			posterior[variable] = values[i]
		
		return posterior


# vars = ["couch", "bookshelf", "tv_set"]
# prior = {"couch": 0.39, "bookshelf": 0.6, "tv_set": 0.01}
# print(util_functions.shannon_entropy(prior))
# bn = ObjBayesNet(vars)
# posterior = bn.infer(prior, {"tv_set": 0})
# print(posterior)
# print(util_functions.shannon_entropy(posterior))
