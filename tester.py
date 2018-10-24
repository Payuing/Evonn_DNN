from __future__ import absolute_import
from __future__ import print_function

print("importing libraries...")
import sys
import time
import math
import csv
import EvoNN
import standardNN
import csv
import warnings
warnings.filterwarnings("ignore")
sys.path.append("/Users/Payu/Desktop/EvoNN_package/EvoNN_DNN") # thrid party's libararies, absolute path

"""Load dataset"""
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_wine

from sklearn.metrics import roc_auc_score # determine which of the used models predicts the classes best

from scipy.stats import wilcoxon # test the null hypothesis that the median of a distribution is equal to some value
import scipy

from sklearn.metrics import mean_squared_error
from math import sqrt

import random
import numpy as np

random.seed(1) # Initialize internal state of the random number generator
np.random.seed(1)

dataset_letter = sys.argv[1] # Choose which dataset to use
my_directory = sys.argv[0].replace("tester.py","") # get directory

########################################################
"""Return RMSE of predicted and ground truth"""
def RMSE(y_predicted, y_true):
	rmse = sqrt(mean_squared_error(y_true, y_predicted))
	return rmse

##############################################################################
def sigmoid(x):
	return 1/(1+np.exp(-x))


##############################################################################
def write_csv_file(filename, the_headers, the_values):
	print("Writing to...",filename)
	the_data_writer, _ = f01_get_default_CSV_writer(filename)


	the_data_writer.writerow(the_headers)
	for i in range(the_values.shape[0]):
		the_data_writer.writerow(the_values[i])

##############################################################################
def f01_get_default_CSV_writer(filename):

	output_file = open(filename, 'w')

	csv.register_dialect(
                    'mydialect',
                    delimiter = ',',
                    quotechar = '"',
                    doublequote = True,
                    skipinitialspace = True,
                    lineterminator = '\n',
                    quoting = csv.QUOTE_MINIMAL)

	thedatawriter = csv.writer(output_file, dialect='mydialect')

	return thedatawriter, output_file
##############################################################################
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

##############################################################################

def ReLU(x):
	y = np.zeros(x.shape)
	y[:] = x[:]
	y[y <= 0] = 0
	return y

##############################################################################
def LReLU(x):

	y = np.zeros(x.shape)
	y[:] = x[:]
	y[x <= 0] = 0.01*x[x <= 0]

	return y
##############################################################################
def two_dec_digs(x):

	y = 100.0*x
	y = np.rint(y)
	y = y / 100.0

	return y
##############################################################################
def to_the_power_of_one_third(x):
	y = np.cbrt(x)

	return y

##############################################################################
def sine_func(x):
	y = np.sin(x)

	return y
##############################################################################
def log_func(x):
	y = np.zeros(x.shape)
	y = np.log(x)
	y[np.isnan(y)] = 0.0

	return y
##############################################################################

def double_threshold(x):
	y = np.zeros(x.shape)
	y[:] = 1.0
	#y[x < -1.0] = 0.01 * x[x < -1.0]
	#y[x > 1.0] = -0.01 * x[x > 1.0]
	y[x < -1.0] = 0.0
	y[x > 1.0] = 0.0

	return y


##############################################################################
def inverse_sigmoid(x):
	return 1.0 - (1/(1+np.exp(-x)))


##############################################################################
def myAUC(y_predicted, y_true):

	#one_d_y_predicted = np.zeros((y_predicted.shape[0], ))
	#one_d_y_true = np.zeros((y_true.shape[0], ))

	one_d_y_predicted = y_predicted[:, 0]
	one_d_y_true = y_true[:, 0]

	#print(one_d_y_predicted.shape)
	#print(one_d_y_true.shape)

	#print(one_d_y_predicted)
	#print(one_d_y_true)

	#for i in range(y_predicted.shape[0]):
	#	one_d_y_true[i] = y_true[i][0]
	#	one_d_y_predicted[i] = y_predicted[i][0]

	result = roc_auc_score(one_d_y_true, one_d_y_predicted)

	#print(result)

	return -1.0 * result
##############################################################################
def multiclass_LOGLOSS(y_predicted, y_true):
	logloss_value = 0.0
	#print(y_predicted, "\n", y_true)

	#exit()
	for i in range(y_predicted.shape[0]):
		for j in range(y_predicted.shape[1]):
			considered_value = min(max(y_predicted[i][j], 1.0E-15), 1.0-1.0E-15)
			logloss_value +=  y_true[i][j]* math.log(considered_value)

	logloss_value *= -(1.0/y_predicted.shape[0])
	return logloss_value


##############################################################################
def linearmax(final_layer_values):

	new_layer_values = np.zeros(final_layer_values.shape)
	for i in range(new_layer_values.shape[0]):
		x = final_layer_values[i]
		min_value = np.amin(x)

		shiftx = x - min_value
		value_sum = np.sum(shiftx)
		new_layer_values[i][:] = shiftx / value_sum


	return new_layer_values

##############################################################################
def softmax(final_layer_values):

	#print(final_layer_values)
	#exit()

	new_layer_values = np.zeros(final_layer_values.shape)
	for i in range(new_layer_values.shape[0]):
		x = final_layer_values[i]
		shiftx = x - np.max(x) # Since we are using an exponent, this is equivalent to dividing. Comes from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
		#shiftx = x

		exponent_layer_values = np.exp(shiftx)
		sum_of_values = np.sum(exponent_layer_values)
		new_layer_values[i] = exponent_layer_values/sum_of_values

	#print(new_layer_values)
	return new_layer_values


##############################################################################

if (dataset_letter == "a"):
	print("Loading iris...")
	dataset = load_iris()
elif (dataset_letter == "b"):
	print("Loading breast cancer...")
	dataset = load_breast_cancer()
elif (dataset_letter == "c"):
	print("Loading diabetes...")
	dataset = load_diabetes()
elif (dataset_letter == "d"):
	print("Loading digits...")
	dataset = load_digits()
elif (dataset_letter == "e"):
	print("Loading breast cancer...")
	dataset = load_breast_cancer()
elif (dataset_letter == "f"):
	print("Loading wine...")
	dataset = load_wine()
else:
	print("Select either a, b, c, d, e")
	exit()


X = dataset.data
Y = dataset.target


for i in range(X.shape[1]):
	largest_value = np.amax(X[:, i])
	smallest_value = np.amin(X[:, i])
	if (largest_value == 0):
		if (smallest_value != 0):
			X[:, i] /= abs(smallest_value)
	else:
		X[:, i] /= max(largest_value, abs(smallest_value))

EVOLVER = True
STANDARD = True

evo_pairs = []
standard_pairs = []

evo_logloss = []
standard_logloss = []

shuffle_number = 20
rep_number = 10

evo_runtimes = []
standard_runtimes = []

evo_headers = []
standard_headers = []
evo_measurements = np.zeros((rep_number, shuffle_number))
standard_measurements = np.zeros((rep_number, shuffle_number))


for a in range(shuffle_number):

	evo_headers.append("shuffle #"+str(a+1))
	standard_headers.append("shuffle #"+str(a+1))
	X, Y = shuffle_in_unison(X, Y)

	sample_size = Y.shape[0]
	class_array = ['a','b','d','e','f']
	if (dataset_letter in class_array):
		myType = 'classification'
		class_number = np.max(Y)+1

		Y_classes = np.zeros((sample_size, class_number))
		for i in range(sample_size):
			j = Y[i]
			Y_classes[i][j] = 1.0
	else:
		myType = 'regression'
		largest_value = np.amax(Y[:])
		smallest_value = np.amin(Y[:])
		Y_classes = Y/max(largest_value, abs(smallest_value))


	print("X is a",X.shape[0],"X",X.shape[1],"matrix")
	if (len(Y_classes.shape) > 1):
		print("Y is a",Y_classes.shape[0],"X",Y_classes.shape[1],"matrix")
	else:
		print("Y is a",Y_classes.shape[0],"X 1 matrix")

	train_index = int(sample_size*0.6)
	validate_index = int(sample_size*0.8)

	X_train, Y_train = X[:train_index], Y_classes[:train_index]
	X_valid, Y_valid = X[train_index:validate_index], Y_classes[train_index:validate_index]
	X_test, Y_test = X[validate_index:], Y_classes[validate_index:]




	for i in range(rep_number):
		#print("\ti = ",i)
		myRep = i+(a*rep_number)
		print("\trep = ",myRep)
		if (EVOLVER == True):

			start_time = time.process_time()




			if (myType == 'classification'):
				final_function = softmax
				#final_function = linearmax
				fitness = multiclass_LOGLOSS
				if (dataset_letter == 'e'):
					fitness = myAUC
			elif (myType == 'regression'):
				#final_function = sigmoid
				final_function = two_dec_digs
				fitness = RMSE

			myEvoNNEvolver = EvoNN.Evolver(	G=10000,
											early_stopping=200,
											MU=50,			# the number of parents
											LAMBDA=50,		# the number of offspring
											P_m=0.01,		#0.1
											P_mf=0.01,		#0.1
											R_m=0.1,		#1.0
											P_c=0.3,		#0.3
											elitism=True,
											tournament_size=2,	#5
											fitness_function=fitness,
											final_activation_function=final_function,
											additional_functions=[LReLU, ReLU], #double_threshold],
											random_state=myRep,
											verbose=0)
			#myEvoNNEvolver.fit(X_train, Y_train)
			myEvoNNEvolver.fit(X_train, Y_train, X_valid, Y_valid)
			if (myType == 'classification'):
				evo_Y_pred_probabilities = myEvoNNEvolver.predict_proba(X_test)
			else:
				evo_Y_pred_probabilities = myEvoNNEvolver.predict(X_test)

			if (myType == 'classification'):
				if (dataset_letter == 'e'):
					evo_my_logloss = myAUC(evo_Y_pred_probabilities, Y_test)
				else:
					evo_my_logloss = multiclass_LOGLOSS(evo_Y_pred_probabilities, Y_test)
			elif (myType == 'regression'):
				evo_my_logloss = RMSE(evo_Y_pred_probabilities,
                                                      Y_test)
			elapsed_time = time.process_time() - start_time
			evo_runtimes.append(elapsed_time)
			evo_logloss.append(evo_my_logloss)

			evo_logloss_arr = np.array(evo_logloss)
			evo_logloss_mean = np.mean(evo_logloss_arr)
			evo_logloss_std = np.std(evo_logloss_arr)

			evo_measurements[i][a] = evo_my_logloss

			print("evo logloss",evo_logloss_mean,"+/-",evo_logloss_std)
			print(evo_logloss_arr)


			evo_avg = np.mean(evo_runtimes)
			evo_SD = np.std(evo_runtimes)

			print("evo NN runtime per cycle = ",evo_avg,"+/-",evo_SD)

		if (STANDARD == True):

			start_time = time.process_time()
			myNN = standardNN.standardNN(	early_stopping=200,
											epoch=10000,
											random_state=myRep,
											type = myType,
											verbose=0)
			myNN.fit(X_train, Y_train, X_valid, Y_valid)
			if (myType == 'classification'):
				standard_Y_pred_probabilities = myNN.predict_proba(X_test)
				if (dataset_letter == 'e'):
					standard_my_logloss = myAUC(standard_Y_pred_probabilities, Y_test)
					if (standard_my_logloss > -0.5):
						standard_my_logloss = -1.0 - standard_my_logloss
				else:
					standard_my_logloss = multiclass_LOGLOSS(standard_Y_pred_probabilities, Y_test)
			else:
				standard_Y_pred = myNN.predict(X_test)
				standard_my_logloss = RMSE(standard_Y_pred, Y_test)

			elapsed_time = time.process_time() - start_time
			standard_runtimes.append(elapsed_time)
			standard_logloss.append(standard_my_logloss)

			standard_logloss_arr = np.array(standard_logloss)
			standard_logloss_mean = np.mean(standard_logloss_arr)
			standard_logloss_std = np.std(standard_logloss_arr)

			standard_measurements[i][a] = standard_my_logloss

			print("standard logloss",standard_logloss_mean,"+/-",standard_logloss_std)
			print(standard_logloss_arr)

			standard_avg = np.mean(standard_runtimes)
			standard_SD = np.std(standard_runtimes)

			print("standard NN runtime per cycle = ",standard_avg,"+/-",standard_SD)

if (STANDARD == True) and (EVOLVER == True):
	W, p_value = scipy.stats.wilcoxon(	x = evo_logloss_arr,
										y = standard_logloss_arr)

	print("p_value is",p_value)


if (STANDARD == True):
	standard_headers = np.array(standard_headers)
	standard_filename = my_directory+dataset_letter+"_standard_results.csv"
	write_csv_file(standard_filename, standard_headers, standard_measurements)



if (EVOLVER == True):
	evo_headers = np.array(evo_headers)
	evo_filename = my_directory+dataset_letter+"_evo_results.csv"
	write_csv_file(evo_filename, evo_headers, evo_measurements)


'''	evo_pairs.append((evo_logloss_mean, evo_logloss_std))
	standard_pairs.append((standard_logloss_mean, standard_logloss_std))


print("evo pairs\t\t\t\tstandard pairs")
for i in range(len(evo_pairs)):
	print(evo_pairs[i][0],"+/-",evo_pairs[i][1],"\t",standard_pairs[i][0],"+/-",standard_pairs[i][1])

if (EVOLVER == True):
	pass
	#

if (STANDARD == True):
	pass


#myEvoNN = EvoNN.EvoNN(feature_number,2)


#print(y)'''
print("Done")
