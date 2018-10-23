from __future__ import absolute_import
from __future__ import print_function

import sys
#import time
#import datetime
#import os
#import os.path
import math
#import re
#import EvoNN

#from lxml import etree

import csv

import warnings
warnings.filterwarnings("ignore")
sys.path.append("/home/bshabash/pythonstuff")

import numpy as np
import random

from sklearn.metrics import mean_squared_error
from math import sqrt




########################################################
def RMSE(y_predicted, y_true):
	rmse = sqrt(mean_squared_error(y_true, y_predicted))
	
	return rmse
	
########################################################
def multiclass_LOGLOSS(y_predicted, y_true):
	logloss_value = 0.0
	
	#print(y_predicted, "\n", y_true, y_predicted)
	
	#exit()
	for i in range(y_predicted.shape[0]):
		for j in range(y_predicted.shape[1]):
			considered_value = min(max(y_predicted[i][j], 1.0E-15), 1.0-1.0E-15)
			logloss_value +=  y_true[i][j]* math.log(considered_value)
			
	logloss_value *= -(1.0/y_predicted.shape[0])
	return logloss_value
	
########################################################
#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))
	
########################################################
def softmax(x):
	
	new_layer_values = np.zeros((x.shape[0], x.shape[1]))
	
	for i in range(new_layer_values.shape[0]):
	
		shiftx = x[i] - np.max(x[i]) # Since we are using an exponent, this is equivalent to dividing. Comes from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
	
		exponent_layer_values = np.exp(shiftx)
		sum_of_values = np.sum(exponent_layer_values)
		
	
		new_layer_values[i] = exponent_layer_values/sum_of_values
	
	return new_layer_values
	


########################################################

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)
	
def derivatives_sigmoid2(x, y):
	return x * (1 - x)

		
def derivatives_softmax(x, Y_true):
	#print(x - Y_true)
	#return x - Y_true
	
	y = softmax(x)
	
	jacobian = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
	
	for k in range(jacobian.shape[0]):
		for j in range(jacobian.shape[1]):
			for i in range(jacobian.shape[2]):
				if (i==j):
					jacobian[k][j][i] = y[k][i]*(1-y[k][i])
				else:
					jacobian[k][j][i] = y[k][i]*y[k][j]
					
	return jacobian
	

derivative_output = derivatives_softmax
########################################################


HIDDEN_LAYER_1_SIZE = 10
	
class standardNN:

	def __init__(	self, 
					early_stopping=None,
					epoch=5000,                                        
					random_state=None,
					type='classification',
					verbose=0):
		

		#Variable initialization
		
		self.epoch=epoch #Setting training iterations
		#self.epoch=3
		
		if (early_stopping is not None):
			self.early_stopping  = early_stopping
		else:
			self.early_stopping = self.epoch
		
		if (random_state is not None):
			np.random.seed(random_state)
			random.seed(random_state)
		self.lr=0.001 #Setting learning rate
		
		self.type = type
		if (self.type != 'regression') and (self.type != 'classification'):
			raise ValueError("Error: type must be \"regression\" or \"classification\"")
		
		if (self.type == 'regression'):
			self.output_function = sigmoid
		elif (self.type == 'classification'):
			self.output_function = softmax
			
		self.verbose = verbose
		
	def get_output(self, X_train):
		
		self.hidden_layer_input1 		= np.dot(X_train,self.wh)
		self.hidden_layer_input 		= self.hidden_layer_input1 #+ self.bh
		self.hiddenlayer_activations	= sigmoid(self.hidden_layer_input)
		self.output_layer_input1 		= np.dot(self.hiddenlayer_activations,self.wout)
		self.output_layer_input 		= self.output_layer_input1 #+ self.bout
		output		 					= self.output_function(self.output_layer_input)
		
		return output
		
		
		
	def fit(self, X_train, Y_train, X_val = None, Y_val = None):
	
		self.inputlayer_neurons = X_train.shape[1] #number of features in data set
		self.hiddenlayer_neurons = HIDDEN_LAYER_1_SIZE #number of hidden layers neurons
		if (self.type == 'regression'):
			self.output_neurons = 1 #number of neurons at output layer
		elif (self.type == 'classification'):
			self.output_neurons = Y_train.shape[1]

		#weight and bias initialization

		self.wh=np.random.uniform(size=(self.inputlayer_neurons,self.hiddenlayer_neurons))
		
		self.bh=np.random.uniform(size=(1,self.hiddenlayer_neurons))
		self.wout=np.random.uniform(size=(self.hiddenlayer_neurons,self.output_neurons))
		self.bout=np.random.uniform(size=(1,self.output_neurons))

		self.best_wh=self.wh
		
		self.best_bh=self.bh
		self.best_wout=self.wout
		self.best_bout=self.bout


		i = 0
		early_stop = False
		val_counter = 0
		best_validation_logloss = float("inf")
		while ((i<self.epoch) and (early_stop == False)):

			#Forward Propogation
			
			
			if (X_val is not None):
				val_output = self.get_output(X_val)
				
				if (self.type == 'regression'):
					myLL = RMSE(val_output, Y_val)
				elif (self.type == 'classification'):
					myLL = multiclass_LOGLOSS(val_output, Y_val)
				
				
				if (myLL < best_validation_logloss):
					best_validation_logloss = myLL
					val_counter = 0
					self.best_wh=self.wh.copy()
		
					self.best_bh=self.bh.copy()
					self.best_wout=self.wout.copy()
					self.best_bout=self.bout.copy()
				else:
					val_counter += 1
					if (val_counter >= self.early_stopping):
						early_stop = True
				
			output = self.get_output(X_train)
			if (self.verbose > 0):
				print("\t\t",i,"\tBackpropegating... validation LL:",myLL,"\tval_counter:",val_counter)
			#Backpropagation
			
			
			#
			#print(output)
			#exit()
			'''E = np.zeros((output.shape[0], output.shape[1], output.shape[1]))
			for k in range(E.shape[0]):
				for i in range(E.shape[1]):
					for j in range(E.shape[2]):
						if (i != j):
							E[k][i][j] = 0.0
						else:
							E[k][i][j] -= Y_train[k][j]/output[k][j]'''
			
			
			slope_output_layer = derivative_output(output, Y_train)
			slope_hidden_layer = derivatives_sigmoid(self.hiddenlayer_activations)
			
			if (self.type == 'regression'):
				
				#output = np.reshape(output, (output.shape[0],))
				#E = Y_train-output
				
				#slope_output_layer = derivatives_sigmoid(output)
				
				#d_output = E * slope_output_layer
				
				
				Y_train = np.reshape(Y_train, (Y_train.shape[0],1))
				#print("Y_train.shape",Y_train.shape)
				#print("output.shape",output.shape)
				E = Y_train-output
				#print("E.shape",E.shape)
				
				slope_output_layer = derivatives_sigmoid(output)
				#print("slope_output_layer",slope_output_layer.shape)
				slope_hidden_layer = derivatives_sigmoid(self.hiddenlayer_activations)
				
				d_output = E * slope_output_layer
				#print("d_output", d_output.shape)
				
				
			elif (self.type == 'classification'):
				d_output = Y_train - output
				
				#d_output
				#d_output = np.zeros(Y_train.shape)
				#d_output = (Y_train * output) - Y_train
				#print(output)
				#print(Y_train)
				#print(d_output)
				#exit()
			#d_output = -(1.0/Y_train.shape[0]) * (output - Y_train)
			#print(d_output)
			#exit()
			
			
			Error_at_hidden_layer = d_output.dot(self.wout.T)
			d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
			self.wout += self.hiddenlayer_activations.T.dot(d_output) *self.lr
			self.bout += np.sum(d_output, axis=0,keepdims=True) *self.lr
			self.wh += X_train.T.dot(d_hiddenlayer) * self.lr
			self.bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) * self.lr
			
			i+= 1
			
	def predict_proba(self, X_test):
    	
		if (self.type == 'regression'):
			raise ValueError("Regression cannot predict probabilities")
		elif (self.type == 'classification'): 
			self.hidden_layer_input1 		= np.dot(X_test,self.best_wh)
			self.hidden_layer_input 		= self.hidden_layer_input1 #+ self.best_bh
			self.hiddenlayer_activations	= sigmoid(self.hidden_layer_input)
			self.output_layer_input1 		= np.dot(self.hiddenlayer_activations, self.best_wout)
			self.output_layer_input 		= self.output_layer_input1 #+ self.best_bout
			Y_predict 						= self.output_function(self.output_layer_input)
		
			return Y_predict
    	
    	
    ######################################################################################
	def predict(self, X_test):
    	
		if (self.type == 'regression'):
			Y_predict = self.get_output(X_test)	
		elif (self.type == 'classification'):
			Y_predict 	= self.predict_proba(X_test)
		
			for i in range(Y_predict.shape[0]):
				max_pos = np.argmax(Y_predict[i])
				Y_predict[i][:] = 0
				Y_predict[i][max_pos] = 1.0
    		
		return Y_predict
