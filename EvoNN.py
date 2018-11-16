from __future__ import absolute_import
from __future__ import print_function

import sys
import math
import csv
import warnings
import numpy as np
import random
import copy
from datetime import datetime
warnings.filterwarnings("ignore") # never print matching warnings
sys.path.append("/Users/Payu/Desktop/EvoNN_package/EvoNN_DNN") #thrid party's libararies, absolute path

# TODO: remove it later
HIDDEN_LAYER_1_SIZE = [10] # node per layer, one layer for now

"""Activation function"""
def sigmoid(x):
	return 1/(1+np.exp(-x))

def tanh(x):
	return np.tanh(x)

"""Loss function"""
def RMSE(y_predicted, y_true):
	y_predicted = y_predicted.reshape((y_predicted.shape[0],))
	return np.sqrt(np.mean((y_predicted - y_true)**2))

"""Return predicted value array"""
def Identity(final_layer_values):
	return final_layer_values[:]


class Evolver:
	def __init__(	self,
					G=10,								# Maximum iteration
					early_stopping=10,					# Minimum iteration
					node_per_layer = [10],				# Number of nodes per layer
					MU=10,								# Number of parents
					LAMBDA=10,							# Number of offspring
					P_m=0.1,							# Weight mutation probability
					P_mf=0.1,							# Function mutation probablity
					R_m=1.0,							# Weight mutation radius
					P_c=0.5,							# Crossover proportion
					P_b=0.01,							# Bias mutation probablity
					R_b=1.0,								# Bias mutation radius
					elitism=True,						# Elitism involves copying a small proportion of the fittest candidates, unchanged, into the next generation.
					tournament_size=2,					# Selecting individuals from a population
					fitness_function=RMSE,
					final_activation_function=Identity,
					additional_functions=[],
					random_state=None,
					verbose=0):

		self.generation_number = G
		self.early_stopping = early_stopping
		self.node_per_layer = node_per_layer
		self.mu = MU
		self.lam = LAMBDA
		self.P_M = P_m
		self.P_MF = P_mf
		self.P_C = P_c
		self.R_M = R_m
		self.P_B = P_b
		self.R_B = R_b
		self.ELITISM = elitism
		self.TOURNAMENT_SIZE = tournament_size
		self.fitness = fitness_function
		self.final_activation = final_activation_function
		self.functions = {0: sigmoid,
                          1: tanh}						# Using a dictionary to select function
		if (random_state is not None):
			time_seconds = int(datetime.now().timestamp())	# Python 3.3+ only
			np.random.seed(random_state + time_seconds)		# add system time to generate random number
			random.seed(random_state + time_seconds)
		self.verbose = verbose

		self.final_population = None
		self.best_individual = None

		key = len(self.functions)										# add additional activation function
		for additional_function in additional_functions:
			self.functions[key] = additional_function
			key += 1

    ######################################################################################
	"""Train the EvoNN"""
	def fit(self, X_train, Y_train, X_val = None, Y_val = None):
		#initialization
		if (self.verbose >= 1):
			print("Input is a {} X {} matrix".format(X_train.shape[0], X_train.shape[1]))
			if (X_val is not None):
				print("Validation is a {} X {} matrix".format(X_val.shape[0], X_val.shape[1]))
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_val = X_val
		self.Y_val = Y_val
		self.best_individual_found = None
		self.feature_number = X_train.shape[1]
		try:
			self.output_number = Y_train.shape[1]	# more than one column
		except IndexError:
			self.output_number = 1
		offspring = []

		population = self.initialize_population() # "mu" used at start to create a populatiion of network
		average_fitness, average_fitness_validate, best_fitness_train, best_fitness_validate, best_individual = self.evaluate_population(population)

		best_fitness_validate_of_all_generations = best_fitness_validate # the smaller the better
		best_individual_validate = best_individual

		#validate_timer = 0
		curr_generation_number = 1
		while ((curr_generation_number < self.generation_number + 1)): # and (self.early_stopping > validate_timer)):
			if (curr_generation_number % 100 == 0):
				print("run for ",curr_generation_number,"generations")
			if (self.verbose >= 1):
				printout_statement = "Generation "+str(curr_generation_number)
				printout_statement += "\tTrain "
				#printout_statement += "average fitness: "+str(average_fitness)
				printout_statement += "\tbest fitness train: "+str(best_fitness_train)
				if (self.X_val is not None):
					printout_statement += "\tValidate "
					#printout+statement += "average fitness: "+str(average_fitness_validate)
					printout_statement += "\tbest fitness: "+str(best_fitness_validate_of_all_generations)
					#printout_statement += "\tvalidate_timer: "+str(validate_timer)
				else:
					print("Please specify validate set.")
					exit()

				print(printout_statement)

			offspring = self.make_offspring(population) # a list of offspring; use lam to generate
			for theOffspring in offspring:
				theOffspring.mutate(self.P_M, self.P_MF, self.R_M, self.P_B, self.R_B)

			population = [] # generate new population

			# Only one best individual
			if (self.ELITISM == True):
				copy_individual = EvoNN.copyIndividual(best_individual)
				population.append(copy_individual)
				init_range = 1
			else:
				init_range = 0

			"""Generate next parent generation"""
			for i in range(init_range, self.mu):
				theOriginal = self.tournament_selection(offspring, self.TOURNAMENT_SIZE)
				copy_individual = EvoNN.copyIndividual(theOriginal)
				population.append(copy_individual)

			average_fitness, average_fitness_validate, best_fitness_train, best_fitness_validate, best_individual = self.evaluate_population(population)

			if (self.X_val is not None):
				if (best_fitness_validate < best_fitness_validate_of_all_generations):
					best_fitness_validate_of_all_generations = best_fitness_validate
					best_individual_validate = copy.deepcopy(best_individual)
					#validate_timer = 0
			else:
				#validate_timer += 1
				print("Please specify validate set.")
				exit()

			curr_generation_number += 1

		self.best_individual = copy.deepcopy(best_individual_validate)
		self.final_population = copy.deepcopy(population)
		if (self.verbose >= 1):
			print(self.best_individual)


	######################################################################################
	""""Predict on test dataset"""
	def predict_proba(self, X_test):
		return self.best_individual.get_output(X_test)

    ######################################################################################
	"""Predict on test dataset"""
	def predict(self, X_test):
		return self.best_individual.get_output(X_test)

    ######################################################################################
	"""NOT USED"""
	def predict_classes(self, X_test):

		Y_predict = self.best_individual.get_output(X_test)
		for i in range(Y_predict.shape[0]):
			max_pos = np.argmax(Y_predict[i])
			Y_predict[i][:] = 0
			Y_predict[i][max_pos] = 1.0

		return Y_predict

    ######################################################################################
	def initialize_population(self):
		if (self.verbose >= 1):
			print("Initializing population...")

		my_population = []
		for i in range(self.mu):
			theIndividual = EvoNN.newIndividual(self.feature_number, self.output_number, self.final_activation, hidden_size = self.node_per_layer, function_dictionary = self.functions)
			my_population.append(theIndividual) # theIndividual is a standalone network
			if (self.verbose >= 1):
				print("\t\t\t {}".format(my_population[i]))

		if (self.verbose >= 1):
			print("Population initialized")
		return my_population
    ######################################################################################
	def evaluate_population(self, the_population):
		if (self.verbose >= 1):
			print("Evaluating population")

		i = 1
		""""Initialize parameters"""
		average_fitness_train = 0.0 # the whole population
		average_fitness_validate = 0.0

		population_count_train = 0
		population_count_validate = 0

		best_fitness_train = the_population[0].fitness
		best_fitness_validate = the_population[0].fitness

		best_individual = the_population[0]

		for individual in the_population:
			Y_predict = individual.get_output(self.X_train)
			fitness_value_train = self.fitness(Y_predict, self.Y_train) # Y_train is a 2d one-hot coding matrix
			individual.fitness = fitness_value_train

			if not (math.isnan(fitness_value_train)):
				average_fitness_train += fitness_value_train
				population_count_train += 1

			"""best_fitness_train: the smaller the better"""
			if (fitness_value_train < best_fitness_train):
				best_fitness_train = fitness_value_train
				best_individual = individual

			if (self.X_val is not None):
				Y_val_predict = individual.get_output(self.X_val)
				fitness_value_validate = self.fitness(Y_val_predict, self.Y_val)
				average_fitness_validate += fitness_value_validate
				population_count_validate += 1
			else:
				print("Please speficy validate dataset")
				exit()

			#individual.fitness = fitness_value_train
			#print(i, "fitness_value", fitness_value)
			#i+= 1

		#if (self.X_val is not None):
		Y_val_predict = best_individual.get_output(self.X_val)
		best_fitness_validate = self.fitness(Y_val_predict, self.Y_val)
		#else:
		#	print("Please speficy validate dataset")
		#	exit()

			#if (fitness_value_validate < best_fitness_validate):
			#	best_fitness_validate = fitness_value_validate

		average_fitness_train /= population_count_train # average fitness for the whole population
		#if (self.X_val is not None):
		average_fitness_validate /= population_count_validate

		if (self.verbose >= 1):
			print("Population evaluated")

		return average_fitness_train, average_fitness_validate, best_fitness_train, best_fitness_validate, best_individual
    ######################################################################################
	def make_offspring(self, the_population):
		if (self.verbose >= 1):
			print("Making offspring")

		offspring_population = []
		for i in range(self.lam):
			offspring_population.append(self.create_offspring(the_population))

		if (self.verbose >= 1):
			print("Made offspring")

		return offspring_population
    ######################################################################################
	def create_offspring(self, the_population):

		random_chance = random.random()
		if (random_chance <= self.P_C): # crossover proportion
			parent1 = self.tournament_selection(the_population)
			parent2 = self.tournament_selection(the_population)
			theIndividual = EvoNN.crossoverIndividual(parent1, parent2)
			return theIndividual
		else:
			original = self.tournament_selection(the_population)
			theIndividual = EvoNN.copyIndividual(original)
			return theIndividual

    ######################################################################################
	""""Tournament selection"""
	def tournament_selection(self, the_population, tournament_size=2):

		population_size = len(the_population)
		the_tournament = []
		for i in range(tournament_size):
			the_tournament.append(the_population[random.randint(0, population_size-1)])

		"""Initialization"""
		best_fitness = the_tournament[0].fitness
		best_individual = the_tournament[0]
		for i in range(1, tournament_size):
			if (the_tournament[i].fitness < best_fitness):
				best_fitness = the_tournament[i].fitness
				best_individual = the_tournament[i]

		return best_individual

    ######################################################################################
    ######################################################################################
	"""
	def __str__(self):
		return_string = ""
		return_string += "G:\t"+str(self.generation_number)+"\n"
		return_string += "MU:\t"+str(self.mu)+"\n"
		return_string += "LAMBDA:\t"+str(self.lam)+"\n"
		return_string += "P_m:\t"+str(self.P_M)+"\n"
		return_string += "P_c:\t"+str(self.P_C)+"\n"
		return_string += "fitness function:\t"+str(self.fitness)+"\n"
		return_string += "final activation function:\t"+str(self.final_activation)+"\n"
		return_string += "functions:\n"
		for function in self.functions:
			return_string += "\t"+str(function)+"\n"

		return return_string
"""



##########################################################################################
class EvoNN:

	default_function_dictionary = {0: sigmoid,
                                   1: tanh}

##########################################################################################
	def __init__(self):
		pass
##########################################################################################
	""""Generate new standalone feedforward network"""
	@classmethod
	def newIndividual(cls, input_size, output_size, final_activation_function, hidden_size=None, function_dictionary = None):

		theIndividual = cls()			#theIndividual is a class
		if (function_dictionary is None):
			theIndividual.function_dictionary = self.default_function_dictionary
		else:
			theIndividual.function_dictionary = function_dictionary
		theIndividual.fitness = float('inf')		# initial fitness is inf
		theIndividual.instance_number = 1			# input data instance
		theIndividual.input_layer = np.zeros((theIndividual.instance_number, input_size)) #+1

		if (hidden_size is None):				#hidden_size is a list
			hidden_size=HIDDEN_LAYER_1_SIZE #+1

		theIndividual.hidden_layer_size = hidden_size # number of layers
		num_hidden_layers = len(theIndividual.hidden_layer_size)
		#theIndividual.hidden_layer = np.zeros((hidden_size)) # need this?

		theIndividual.hidden_layer = [] # a list of numpy 1d array
		theIndividual.hidden_layer_bias = [] # a list of numpy 1d array
		for node_size in hidden_size: # hidden_size is a list
			theIndividual.hidden_layer.append(np.zeros((node_size))) # Need this?
			theIndividual.hidden_layer_bias.append(np.random.uniform(size=(node_size)))

		#theIndividual.hidden_layer_functions = np.random.randint(len(theIndividual.function_dictionary.keys()),
        #                                                size=hidden_size)

		theIndividual.hidden_layer_functions = [] # a list of numpy 1d array
		for node_size in hidden_size:
			theIndividual.hidden_layer_functions.append(np.random.randint(
			len(theIndividual.function_dictionary.keys()), size=node_size
			))

		theIndividual.output_layer = np.zeros((output_size))
		theIndividual.final_activation = final_activation_function # softmax, probability function

		#theIndividual.hidden_layer_size = len(hidden_size) # number of layers
		theIndividual.input_to_hidden_matrix = np.random.uniform(size=(	input_size, hidden_size[0]))

		if (num_hidden_layers > 1):
			theIndividual.hidden_to_hidden_matrix = []
			for curr_layer in range(num_hidden_layers - 1):
				theIndividual.hidden_to_hidden_matrix.append(np.random.uniform(size=(hidden_size[curr_layer], hidden_size[curr_layer + 1])))

		theIndividual.hidden_to_output_matrix = np.random.uniform(size=( hidden_size[-1], output_size))

		return theIndividual

##########################################################################################
	@classmethod
	def crossoverIndividual(cls, individual1, individual2):

		theIndividual = cls() # the offspring individual
		theIndividual.function_dictionary = individual1.function_dictionary

		input_size = individual1.input_to_hidden_matrix.shape[0]
		output_size = individual1.hidden_to_output_matrix.shape[1]

		theIndividual.fitness = float('inf')
		theIndividual.instance_number = 1
		theIndividual.input_layer = np.zeros((theIndividual.instance_number, input_size))

		#hidden_size=individual1.input_to_hidden_matrix.shape[1]
		hidden_size = individual1.hidden_layer_size # a list array
		num_hidden_layers = len(hidden_size)

		# generate offspring arch
		theIndividual.hidden_layer_size  = copy.deepcopy(hidden_size)
		theIndividual.hidden_layer = []
		theIndividual.hidden_layer_bias = []
		theIndividual.hidden_layer_functions = []
		for node_size in hidden_size:
			theIndividual.hidden_layer.append(np.zeros(node_size)) # need this?
			theIndividual.hidden_layer_bias.append(np.zeros(node_size))
			theIndividual.hidden_layer_functions.append(np.zeros(node_size))

		#theIndividual.hidden_layer = np.zeros((hidden_size))
		#theIndividual.hidden_layer_functions = np.zeros((hidden_size))

		theIndividual.output_layer = np.zeros((individual1.output_layer.shape[0]))
		theIndividual.final_activation = individual1.final_activation

		"""crossover activation function and bias"""
		for layer in range(num_hidden_layers):
			# crossover activation function
			probablity_matrix = np.random.uniform(size=(hidden_size[layer]))
			theIndividual.hidden_layer_functions[layer][probablity_matrix <= 0.5] = individual1.hidden_layer_functions[layer][probablity_matrix <= 0.5]
			theIndividual.hidden_layer_functions[layer][probablity_matrix > 0.5] = individual2.hidden_layer_functions[layer][probablity_matrix > 0.5]

			# crossover bias
			probablity_matrix = np.random.uniform(size=(hidden_size[layer]))
			theIndividual.hidden_layer_bias[layer][probablity_matrix <= 0.5] = individual1.hidden_layer_bias[layer][probablity_matrix <= 0.5]
			theIndividual.hidden_layer_bias[layer][probablity_matrix > 0.5] = individual2.hidden_layer_bias[layer][probablity_matrix > 0.5]

		"""crossover weight matrix"""
		# input to hidden matrix
		theIndividual.input_to_hidden_matrix = np.zeros((input_size, hidden_size[0]))
		probablity_matrix = np.random.uniform(size=(input_size, hidden_size[0]))

		theIndividual.input_to_hidden_matrix[probablity_matrix <= 0.5] = individual1.input_to_hidden_matrix[probablity_matrix <= 0.5]
		theIndividual.input_to_hidden_matrix[probablity_matrix > 0.5] = individual2.input_to_hidden_matrix[probablity_matrix > 0.5]

		# hidden to hidden matrix
		if (num_hidden_layers > 1):
			theIndividual.hidden_to_hidden_matrix = []
			for curr_layer in range(num_hidden_layers - 1):
				new_hidden_to_hidden_matrix = np.zeros((hidden_size[curr_layer], hidden_size[curr_layer + 1]))
				probablity_matrix = np.random.uniform(size=(hidden_size[curr_layer], hidden_size[curr_layer + 1]))

				new_hidden_to_hidden_matrix[probablity_matrix <= 0.5] = individual1.hidden_to_hidden_matrix[curr_layer][probablity_matrix <= 0.5]
				new_hidden_to_hidden_matrix[probablity_matrix > 0.5] = individual2.hidden_to_hidden_matrix[curr_layer][probablity_matrix > 0.5]

				theIndividual.hidden_to_hidden_matrix.append(new_hidden_to_hidden_matrix)

		# hidden to output matrix
		theIndividual.hidden_to_output_matrix = np.zeros((hidden_size[-1], output_size))
		probablity_matrix = np.random.uniform(size=((hidden_size[-1], output_size)))

		theIndividual.hidden_to_output_matrix[probablity_matrix <= 0.5] = individual1.hidden_to_output_matrix[probablity_matrix <= 0.5]
		theIndividual.hidden_to_output_matrix[probablity_matrix > 0.5] = individual2.hidden_to_output_matrix[probablity_matrix > 0.5]

		print("Offspring has {} layers".format(theIndividual.hidden_layer_size))
		return theIndividual

##########################################################################################
	""""Deep copy individual"""
	@classmethod
	def copyIndividual(cls, theOriginal):

		theIndividual = cls()
		theIndividual.function_dictionary = theOriginal.function_dictionary

		input_size = theOriginal.input_to_hidden_matrix.shape[0]
		output_size = theOriginal.hidden_to_output_matrix.shape[1]

		theIndividual.fitness = float('inf')
		theIndividual.instance_number = 1
		theIndividual.input_layer = np.zeros((theIndividual.instance_number, input_size))

		theIndividual.hidden_layer_size = copy.deepcopy(theOriginal.hidden_layer_size)

		# deep copy bias and activation function
		theIndividual.hidden_layer = copy.deepcopy(theOriginal.hidden_layer) # a list
		theIndividual.hidden_layer_bias = copy.deepcopy(theOriginal.hidden_layer_bias)
		theIndividual.hidden_layer_functions = copy.deepcopy(theOriginal.hidden_layer_functions)

		theIndividual.output_layer = np.zeros((output_size))
		theIndividual.final_activation = theOriginal.final_activation

		# deep copy weight matrix
		theIndividual.input_to_hidden_matrix = copy.deepcopy(theOriginal.input_to_hidden_matrix)
		if (len(theIndividual.hidden_layer_size) > 1):
			theIndividual.hidden_to_hidden_matrix = copy.deepcopy(theOriginal.hidden_to_hidden_matrix)
		theIndividual.hidden_to_output_matrix = copy.deepcopy(theOriginal.hidden_to_output_matrix)

		return theIndividual

##########################################################################################
	def mutate_matrix(self, the_matrix, probablity, radius):

		probablity_matrix = np.random.uniform(size=(the_matrix.shape))
		mutation_matrix = np.random.uniform(low = -radius, high=radius, size=(the_matrix.shape))
		the_matrix[probablity_matrix <= probablity] += mutation_matrix[probablity_matrix <= probablity]

		return the_matrix

##########################################################################################
	def mutate(self, P_m, P_mf, R_m, P_b, R_b):

		input_size = self.input_layer.shape[1]
		hidden_size= self.hidden_layer_size # a list
		num_hidden_layers = len(self.hidden_layer_size)
		output_size = self.hidden_to_output_matrix.shape[1]

		""""Mutate input to hidden matrix"""
		self.input_to_hidden_matrix = self.mutate_matrix(self.input_to_hidden_matrix, P_m, R_m)

		""""Mutate activation function and bias"""
		function_number = len(self.function_dictionary.keys())

		for layer in range(num_hidden_layers):
			# mutate activation function
			probablity_matrix = np.random.uniform(size=(hidden_size[layer]))
			function_mutation_matrix = np.random.randint(0, function_number - 1,size=(hidden_size[layer]))
			self.hidden_layer_functions[layer][probablity_matrix <= P_mf] = function_mutation_matrix[probablity_matrix <= P_mf]

			# mutate bias
			self.hidden_layer_bias[layer] = self.mutate_matrix(self.hidden_layer_bias[layer], P_b, R_b)

		"""Mutate hidden to hidden matrix"""
		if (num_hidden_layers > 1):
			for layer in range(num_hidden_layers - 1):
				self.hidden_to_hidden_matrix[layer] = self.mutate_matrix(self.hidden_to_hidden_matrix[layer], P_m, R_m)

		"""Mutate hidden to output matrix"""
		self.hidden_to_output_matrix = self.mutate_matrix(self.hidden_to_output_matrix, P_m, R_m)

##########################################################################################
	"""Output is a 2d (sample_size, classification_number) array"""
	def get_output(self, X_train):

		#self.hidden_layer = np.dot(X_train,self.input_to_hidden_matrix)
		#hidden_layer_input = self.hidden_layer #+ self.bh
		sample_size = X_train.shape[0]
		hidden_layer_input = np.dot(X_train, self.input_to_hidden_matrix) + np.tile(self.hidden_layer_bias[0], (sample_size, 1)) # y = wx+b

		for i in range(hidden_layer_input.shape[1]): # z = f(wx+b)
			functionIndex = self.hidden_layer_functions[0][i]
			myFunction = self.function_dictionary[functionIndex]
			hidden_layer_input[:, i] = myFunction(hidden_layer_input[:, i])

		hidden_layer_matrix = np.copy(hidden_layer_input) # deep copy
		if (len(self.hidden_layer_size) > 1):
			for i in range(len(self.hidden_layer_size) - 1): # aw+b
				hidden_layer_matrix = np.dot(hidden_layer_matrix, self.hidden_to_hidden_matrix[i]) + np.tile(self.hidden_layer_bias[i+1],(sample_size, 1)) # y = wx+b
				# z = f(wx+b)
				for j in range(hidden_layer_matrix.shape[1]):
					functionIndex = self.hidden_layer_functions[i+1][j]
					myFunction = self.function_dictionary[functionIndex]
					hidden_layer_matrix[:, j] = myFunction(hidden_layer_matrix[:, j])

		#for i in range(self.hidden_layer.shape[0]):
		#for j in range(self.hidden_layer.shape[1]):
		#	functionIndex 			= self.hidden_layer_functions[j]
		#	myFunction 				= self.function_dictionary[functionIndex]
		#	self.hidden_layer[:, j]	= myFunction(hidden_layer_input[ :, j])

		output_layer_input = np.dot(hidden_layer_matrix, self.hidden_to_output_matrix)
		# output_layer_input = self.output_layer #+ self.bout

		output = self.final_activation(output_layer_input)

		return output
##########################################################################################
	#""""""
	#def calculate_output_values(self, X):

	#	output = self.get_output(X)
	#	return output

		#output = np.zeros((X.shape[0], self.output_layer.shape[0]))
		#for j in range(X.shape[0]):
			#print(j+1)
			#print(str(int(100*j/X.shape[0]))+"%")
		#	self.input_layer[0, :] = X[j, :]
			#self.input_layer[0, :-1] = X[j, :]
			#self.input_layer[0][-1] = 1.0
			#print("\tinput", self.input_layer[0, :])
		#	hidden_layer_input = np.dot(self.input_layer, self.input_to_hidden_matrix)


		#	for i in range(hidden_layer_input.shape[1]): #-1
		#		myFunction = self.function_dictionary[self.hidden_layer_functions[i]]

		#		self.hidden_layer[i] = myFunction(hidden_layer_input[0][i])
			#print("\thidden", self.hidden_layer)

			#self.hidden_layer[-1] = 1.0


		#	self.output_layer = np.dot(self.hidden_layer, self.hidden_to_output_matrix)
		#	output[j, :] = self.output_layer[:]
		#	output[j, :] = self.final_activation(output[j, :])
			#print("\thidden->output",self.hidden_to_output_matrix)
			#print("\toutput", output[j, :])
			#print()

		#return output

##########################################################################################
	"""A readable string presentation of EvoNN"""
	#def __str__(self):

	#	my_string = ""
		#my_string += "input: \t\t"+str(self.input_layer)+"\n"
	#	my_string += "input_to_hidden_weights: \t"+str(self.input_to_hidden_matrix)+"\n"
		#my_string += "hidden: \t"+str(self.hidden_layer)+"\n"
	#	my_string += "hidden functions: \t"+str(self.hidden_layer_functions)+"\n"
	#	my_string += "hidden_to_output_weights: \t"+str(self.hidden_to_output_matrix)+"\n"
		#my_string += "output: \t"+str(self.output_layer)+"\n"
	#	my_string += "fitness: \t"+str(self.fitness)

	#	return my_string
