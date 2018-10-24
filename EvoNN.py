from __future__ import absolute_import
from __future__ import print_function

import sys
import math
import csv
import warnings
warnings.filterwarnings("ignore") # never print matching warnings
sys.path.append("/Users/Payu/Desktop/EvoNN_package/EvoNN_DNN") #thrid party's libararies, absolute path

import numpy as np
import random


HIDDEN_LAYER_1_SIZE = 10 # node per layer, one layer for now

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
					G=10,
					early_stopping=10,
					MU=10,
					LAMBDA=10,
					P_m=0.1,
					P_mf=0.1,
					R_m=1.0,
					P_c=0.5,
					elitism=True,
					tournament_size=2,
					fitness_function=RMSE,
					final_activation_function=Identity,
					additional_functions=[],
					random_state=None,
					verbose=0):

		self.generation_number = G
		self.early_stopping = early_stopping
		self.mu = MU
		self.lam = LAMBDA
		self.P_M = P_m
		self.P_MF = P_mf
		self.P_C = P_c
		self.R_M = R_m
		self.ELITISM = elitism
		self.TOURNAMENT_SIZE = tournament_size
		self.fitness = fitness_function
		self.final_activation = final_activation_function
		self.functions = {0: sigmoid,
                          1: tanh}
		#self.functions = {0: sigmoid}
		if (random_state is not None):
			np.random.seed(random_state)
			random.seed(random_state)
		self.verbose = verbose

		self.final_population = None
		self.best_individual = None

		key = 2
		for additional_function in additional_functions:
			#self.functions.append(additional_function)
			self.functions[key] = additional_function
			key += 1


    ######################################################################################
	def fit(self, X_train, Y_train, X_val = None, Y_val = None):

		if (self.verbose > 1):
			print("Input is a "+str(X_train.shape[0])+"X"+str(X_train.shape[1])+" matrix")
			if (X_val is not None):
				print("Validation is a "+str(X_val.shape[0])+"X"+str(X_val.shape[1])+" matrix")
		self.X_train = X_train
		self.Y_train = Y_train
		self.X_val = X_val
		self.Y_val = Y_val

		self.best_individual_found = None
		offspring = []

		self.feature_number = X_train.shape[1]
		try:
			self.output_number = Y_train.shape[1]
		except IndexError:
			self.output_number = 1




		population = self.initialize_population()
		average_fitness, average_fitness_validate, best_fitness, best_fitness_validate, best_individual = self.evaluate_population(population)

		validate_timer = 0
		best_fitness_validate_of_all_generations = best_fitness_validate
		best_individual_validate = best_individual

		g = 1
		while ((g<self.generation_number+1) and (self.early_stopping > validate_timer)):
			if (self.verbose > 0):
				printout_statement = "Generation "+str(g)
				printout_statement += "\tTrain "
				#printout_statement += "average fitness: "+str(average_fitness)
				printout_statement += "\tbest fitness: "+str(best_fitness)
				if (self.X_val is not None):
					printout_statement += "\tValidate "
					#printout+statement += "average fitness: "+str(average_fitness_validate)
					printout_statement += "\tbest fitness: "+str(best_fitness_validate_of_all_generations)
					printout_statement += "\tvalidate_timer: "+str(validate_timer)

				print(printout_statement)

			offspring = self.make_offspring(population)
			for theOffspring in offspring:
				theOffspring.mutate(self.P_M, self.P_MF, self.R_M)


			population = []

			if (self.ELITISM == True):
				copy_individual = EvoNN.copyIndividual(best_individual)
				population.append(copy_individual)
				init_range = 1
			else:
				init_range = 0
			for i in range(init_range, self.mu):
				#random_index = random.randint(0, self.lam-1)
				#theOriginal = offspring[random_index]

				theOriginal = self.tournament_selection(offspring, self.TOURNAMENT_SIZE)

				copy_individual = EvoNN.copyIndividual(theOriginal)
				population.append(copy_individual)

			average_fitness, average_fitness_validate, best_fitness, best_fitness_validate, best_individual = self.evaluate_population(population)
			if (self.X_val is not None):
				if (best_fitness_validate < best_fitness_validate_of_all_generations):
					best_fitness_validate_of_all_generations = best_fitness_validate
					best_individual_validate = best_individual
					validate_timer = 0
				else:
					validate_timer += 1

			if (self.X_val is not None):
				self.best_individual = best_individual_validate
			else:
				self.best_individual = best_individual
			self.final_population = population
			g += 1


		#self.best_individual = best_individual_validate
		#self.final_popualtion = population

		if (self.verbose > 0):
			print(self.best_individual)

			#exit()

		print("ran for ",g,"generations")


	######################################################################################
	def predict_proba(self, X_test):

		return self.best_individual.calculate_output_values(X_test)


    ######################################################################################
	def predict(self, X_test):
		return self.best_individual.calculate_output_values(X_test)

    ######################################################################################
	def predict_classes(self, X_test):

		Y_predict = self.best_individual.calculate_output_values(X_test)
		for i in range(Y_predict.shape[0]):
			max_pos = np.argmax(Y_predict[i])
			Y_predict[i][:] = 0
			Y_predict[i][max_pos] = 1.0

		return Y_predict

    ######################################################################################
	def initialize_population(self):
		if (self.verbose > 1):
			print("Initializing population")

		my_population = []
		for i in range(self.mu):
			theIndividual = EvoNN.newIndividual(self.feature_number, self.output_number, self.final_activation, function_dictionary = self.functions)
			my_population.append(theIndividual)
			#print("\t\t\t"+str(my_population[i]))

		if (self.verbose > 1):
			print("Population initialized")
		return my_population
    ######################################################################################
	def evaluate_population(self, the_population):
		if (self.verbose > 1):
			print("Evaluating population")

		i = 1
		average_fitness_train = 0.0
		average_fitness_validate = 0.0

		population_count_train = 0
		population_count_validate = 0

		best_fitness_train = the_population[0].fitness
		best_fitness_validate = the_population[0].fitness

		best_individual = the_population[0]




		for individual in the_population:
			Y_predict = individual.calculate_output_values(self.X_train)
			fitness_value_train = self.fitness(Y_predict, self.Y_train)

			if not (math.isnan(fitness_value_train)):
				average_fitness_train += fitness_value_train
				population_count_train += 1


			if (fitness_value_train < best_fitness_train):
				best_fitness_train = fitness_value_train
				best_individual = individual



			if (self.X_val is not None):
				Y_val_predict = individual.calculate_output_values(self.X_val)
				fitness_value_validate = self.fitness(Y_val_predict, self.Y_val)


				average_fitness_validate += fitness_value_validate
				population_count_validate += 1

			individual.fitness = fitness_value_train
			#print(i, "fitness_value", fitness_value)
			i+= 1

		if (self.X_val is not None):
			Y_val_predict = best_individual.calculate_output_values(self.X_val)
			best_fitness_validate = self.fitness(Y_val_predict, self.Y_val)

			#if (fitness_value_validate < best_fitness_validate):
			#	best_fitness_validate = fitness_value_validate





		average_fitness_train /= population_count_train
		if (self.X_val is not None):
			average_fitness_validate /= population_count_validate

		if (self.verbose > 1):
			print("Population evaluated")

		return average_fitness_train, average_fitness_validate, best_fitness_train, best_fitness_validate, best_individual
    ######################################################################################
	def make_offspring(self, the_population):
		if (self.verbose > 1):
			print("Making offspring")

		offspring_population = []
		for i in range(self.lam):
			offspring_population.append(self.create_offspring(the_population))

		if (self.verbose > 1):
			print("Made offspring")

		return offspring_population
    ######################################################################################
	def create_offspring(self, the_population):

		random_chance = random.random()
		if (random_chance <= self.P_C):
			parent1 = self.tournament_selection(the_population)
			parent2 = self.tournament_selection(the_population)
			theIndividual = EvoNN.crossoverIndividual(parent1, parent2)
			return theIndividual
		else:
			original = self.tournament_selection(the_population)
			theIndividual = EvoNN.copyIndividual(original)
			return theIndividual


		#print(random_chance)
		#exit()

		#for i in range(len(the_population)):

    ######################################################################################
	def tournament_selection(self, the_population, tournament_size=2):

		population_size = len(the_population)

		the_tournament = []
		for i in range(tournament_size):
			the_tournament.append(the_population[random.randint(0, population_size-1)])


		best_fitness = the_tournament[0].fitness
		best_individual = the_tournament[0]
		for i in range(tournament_size):
			if (the_tournament[i].fitness < best_fitness):
				best_fitness = the_tournament[i].fitness
				best_individual = the_tournament[i]

		return best_individual

    ######################################################################################
    ######################################################################################

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




##########################################################################################
class EvoNN:

	default_function_dictionary = {0: sigmoid,
                                   1: tanh}

##########################################################################################
	def __init__(self):
		pass
##########################################################################################
	@classmethod
	def newIndividual(cls, input_size, output_size, final_activation_function, hidden_size=None, function_dictionary = None):

		theIndividual = cls()
		if (function_dictionary is None):
			theIndividual.function_dictionary = self.default_function_dictionary
		else:
			theIndividual.function_dictionary = function_dictionary
		theIndividual.fitness = float('inf')
		theIndividual.instance_number = 1
		theIndividual.input_layer = np.zeros((theIndividual.instance_number, input_size)) #+1

		if (hidden_size is None):
			#hidden_size = random.randint(input_size, 2*input_size)
			hidden_size=HIDDEN_LAYER_1_SIZE #+1
			#hidden_size = max(input_size//2, 2)+1





		theIndividual.hidden_layer = np.zeros((hidden_size))
		theIndividual.hidden_layer_functions = np.random.randint(len(theIndividual.function_dictionary.keys()),
                                                        size=hidden_size)
		#print(self.hidden_layer_functions)

		theIndividual.output_layer = np.zeros((output_size))
		theIndividual.final_activation = final_activation_function

		theIndividual.input_to_hidden_matrix = np.random.uniform(size=(	input_size, #+1
																	hidden_size)) #-1
		#theIndividual.input_to_hidden_matrix *= 2.0
		#theIndividual.input_to_hidden_matrix -= 1.0

		theIndividual.hidden_to_output_matrix = np.random.uniform(size=(	hidden_size,
																	output_size))

		#theIndividual.hidden_to_output_matrix *= 2.0
		#theIndividual.hidden_to_output_matrix -= 1.0

		#self.hidden_layer = np.dot(self.input_layer, self.input_to_hidden_matrix)
		#self.output_layer = np.dot(self.hidden_layer, self.hidden_to_output_matrix)

		return theIndividual

##########################################################################################
	@classmethod
	def crossoverIndividual(cls, individual1, individual2):

		theIndividual = cls()
		theIndividual.function_dictionary = individual1.function_dictionary

		input_size = individual1.input_to_hidden_matrix.shape[0]
		output_size = individual1.hidden_to_output_matrix.shape[1]

		theIndividual.fitness = float('inf')
		theIndividual.instance_number = 1
		theIndividual.input_layer = np.zeros((theIndividual.instance_number, input_size))

		hidden_size=individual1.input_to_hidden_matrix.shape[1]

		theIndividual.hidden_layer = np.zeros((hidden_size))
		theIndividual.hidden_layer_functions = np.zeros((hidden_size))

		for i in range(hidden_size):
			theIndividual.hidden_layer_functions[i] = individual1.hidden_layer_functions[i]
			if (random.random() > 0.5):
				theIndividual.hidden_layer_functions[i] = individual2.hidden_layer_functions[i]


		theIndividual.output_layer = np.zeros((individual1.output_layer.shape[0]))
		theIndividual.final_activation = individual1.final_activation

		theIndividual.input_to_hidden_matrix = np.zeros((input_size, hidden_size)) #-1

#		print(theIndividual.input_to_hidden_matrix[0],"\n")
#		print(individual1.input_to_hidden_matrix[0],"\n")
#		print(individual2.input_to_hidden_matrix[0],"\n")

		chance_matrix = np.random.uniform(size=((input_size, hidden_size)))


		theIndividual.input_to_hidden_matrix[chance_matrix <= 0.5] = individual1.input_to_hidden_matrix[chance_matrix <= 0.5]
		theIndividual.input_to_hidden_matrix[chance_matrix > 0.5] = individual2.input_to_hidden_matrix[chance_matrix > 0.5]

		#for i in range(input_size):
		#	for j in range(hidden_size): #-1
		#		theIndividual.input_to_hidden_matrix[i][j] = individual1.input_to_hidden_matrix[i][j]
		#		if (random.random() > 0.5):
		#			theIndividual.input_to_hidden_matrix[i][j] = individual2.input_to_hidden_matrix[i][j]

#		print(theIndividual.input_to_hidden_matrix[0],"\n")
#		print("CROSS: populate transition matrix1")
#		exit()

		theIndividual.hidden_to_output_matrix = np.zeros((hidden_size, output_size))

		chance_matrix = np.random.uniform(size=((hidden_size, output_size)))


		theIndividual.hidden_to_output_matrix[chance_matrix <= 0.5] = individual1.hidden_to_output_matrix[chance_matrix <= 0.5]
		theIndividual.hidden_to_output_matrix[chance_matrix > 0.5] = individual2.hidden_to_output_matrix[chance_matrix > 0.5]

		#for i in range(hidden_size):
		#	for j in range(output_size):
		#		theIndividual.hidden_to_output_matrix[i][j] = individual1.hidden_to_output_matrix[i][j]
		#		if (random.random() > 0.5):
		#			theIndividual.hidden_to_output_matrix[i][j] = individual2.hidden_to_output_matrix[i][j]


#		print("CROSS: populate transition matrix2")
#		exit()

		return theIndividual

		#self.hidden_layer = np.dot(self.input_layer, self.input_to_hidden_matrix)
		#self.output_layer = np.dot(self.hidden_layer, self.hidden_to_output_matrix)
##########################################################################################
	@classmethod
	def copyIndividual(cls, theOriginal):

		theIndividual = cls()
		theIndividual.function_dictionary = theOriginal.function_dictionary

		input_size = theOriginal.input_to_hidden_matrix.shape[0]
		output_size = theOriginal.hidden_to_output_matrix.shape[1]

		theIndividual.fitness = float('inf')
		theIndividual.instance_number = 1
		theIndividual.input_layer = np.zeros((theIndividual.instance_number, input_size))

		hidden_size=theOriginal.input_to_hidden_matrix.shape[1]

		theIndividual.hidden_layer = np.zeros((hidden_size))
		theIndividual.hidden_layer_functions = np.zeros((hidden_size))

		for i in range(hidden_size): #-1
			theIndividual.hidden_layer_functions[i] = theOriginal.hidden_layer_functions[i]


		theIndividual.output_layer = np.zeros((theOriginal.output_layer.shape[0]))
		theIndividual.final_activation = theOriginal.final_activation

		theIndividual.input_to_hidden_matrix = np.zeros((input_size, hidden_size))

#		print(theIndividual.input_to_hidden_matrix[0],"\n")
#		print(theOriginal.input_to_hidden_matrix[0],"\n")

		theIndividual.input_to_hidden_matrix[:, :] = theOriginal.input_to_hidden_matrix[:, :]

		#for i in range(input_size):
		#	for j in range(hidden_size): #-1
		#		theIndividual.input_to_hidden_matrix[i][j] = theOriginal.input_to_hidden_matrix[i][j]

#		print(theIndividual.input_to_hidden_matrix[0],"\n")
#		print("COPY: populate transition matrix1")
#		exit()

		theIndividual.hidden_to_output_matrix = np.zeros((hidden_size, output_size))
		theIndividual.hidden_to_output_matrix[:, :] = theOriginal.hidden_to_output_matrix[:, :]

		#for i in range(hidden_size):
		#	for j in range(output_size):
		#		theIndividual.hidden_to_output_matrix[i][j] = theOriginal.hidden_to_output_matrix[i][j]

#		print("COPY: populate transition matrix2")
#		exit()

		return theIndividual

		#self.hidden_layer = np.dot(self.input_layer, self.input_to_hidden_matrix)
		#self.output_layer = np.dot(self.hidden_layer, self.hidden_to_output_matrix)



##########################################################################################
	def mutate_matrix(self, the_matrix, P_m, p_r):

		chance_matrix = np.random.uniform(size=(the_matrix.shape))
		mutation_matrix = np.random.uniform(low = -p_r, high=p_r, size=(the_matrix.shape))

		the_matrix[chance_matrix <= P_m] += mutation_matrix[chance_matrix <= P_m]

		return the_matrix
##########################################################################################
	def mutate(self, P_m, P_mf, p_r):


		input_size = self.input_layer.shape[1]
		hidden_size= self.input_to_hidden_matrix.shape[1]
		output_size = self.hidden_to_output_matrix.shape[1]



		self.input_to_hidden_matrix = self.mutate_matrix(self.input_to_hidden_matrix, P_m, p_r)

		#for i in range(input_size):
		#	for j in range(hidden_size): #-1
		#		if (random.random() <= P_m):
		#			self.input_to_hidden_matrix[i][j] += random.uniform(-p_r, p_r)

		function_number = len(self.function_dictionary.keys())

		for i in range(hidden_size):
			if (random.random() <= P_mf):
				self.hidden_layer_functions[i] = random.randint(0, function_number-1)


		self.hidden_to_output_matrix = self.mutate_matrix(self.hidden_to_output_matrix, P_m, p_r)

		#for i in range(hidden_size):
		#	for j in range(output_size):
		#		if (random.random() <= P_m):
		#			self.hidden_to_output_matrix[i][j] += random.uniform(-p_r, p_r)
##########################################################################################
	def get_output(self, X_train):


		self.hidden_layer 				= np.dot(X_train,self.input_to_hidden_matrix)
		hidden_layer_input 				= self.hidden_layer #+ self.bh


		#for i in range(self.hidden_layer.shape[0]):
		for j in range(self.hidden_layer.shape[1]):
			functionIndex 			= self.hidden_layer_functions[j]
			myFunction 				= self.function_dictionary[functionIndex]
			self.hidden_layer[:, j]	= myFunction(hidden_layer_input[ :, j])


		#print("output_layer",self.output_layer.shape)
		self.output_layer				= np.dot(self.hidden_layer,self.hidden_to_output_matrix)
		#print("output_layer",self.output_layer.shape)
		#exit()

		output_layer_input 				= self.output_layer #+ self.bout

		#print("output_layer_input",output_layer_input.shape)
		#exit()
		output		 					= self.final_activation(output_layer_input)
		#exit()
		return output
##########################################################################################
	def calculate_output_values(self, X):


		output = self.get_output(X)
		return output

		output = np.zeros((X.shape[0], self.output_layer.shape[0]))
		for j in range(X.shape[0]):
			#print(j+1)
			#print(str(int(100*j/X.shape[0]))+"%")
			self.input_layer[0, :] = X[j, :]
			#self.input_layer[0, :-1] = X[j, :]
			#self.input_layer[0][-1] = 1.0
			#print("\tinput", self.input_layer[0, :])
			hidden_layer_input = np.dot(self.input_layer, self.input_to_hidden_matrix)


			for i in range(hidden_layer_input.shape[1]): #-1
				myFunction = self.function_dictionary[self.hidden_layer_functions[i]]

				self.hidden_layer[i] = myFunction(hidden_layer_input[0][i])
			#print("\thidden", self.hidden_layer)

			#self.hidden_layer[-1] = 1.0


			self.output_layer = np.dot(self.hidden_layer, self.hidden_to_output_matrix)
			output[j, :] = self.output_layer[:]
			output[j, :] = self.final_activation(output[j, :])
			#print("\thidden->output",self.hidden_to_output_matrix)
			#print("\toutput", output[j, :])
			#print()

		return output




##########################################################################################

	def __str__(self):

		my_string = ""
		#my_string += "input: \t\t"+str(self.input_layer)+"\n"
		my_string += "input_to_hidden_weights: \t"+str(self.input_to_hidden_matrix)+"\n"
		#my_string += "hidden: \t"+str(self.hidden_layer)+"\n"
		my_string += "hidden functions: \t"+str(self.hidden_layer_functions)+"\n"
		my_string += "hidden_to_output_weights: \t"+str(self.hidden_to_output_matrix)+"\n"
		#my_string += "output: \t"+str(self.output_layer)+"\n"
		my_string += "fitness: \t"+str(self.fitness)

		return my_string
