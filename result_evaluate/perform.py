import csv
import copy
import numpy as np
from scipy.stats import wilcoxon
fp_evo = "./EvoNN_DNN/ten_layer_lab/e_evo_results.csv"
fp_fnn = "./EvoNN_DNN/ten_layer_lab/e_standard_results.csv"
with open(fp_fnn, 'r') as f:
    reader = csv.reader(f)
    data_list = list(reader)

# into one list
row = len(data_list)
#col = len(data_list[1])
logloss = []
for i in range(row):
    col = len(data_list[i])
    for j in range(col):
        logloss.append(float(data_list[i][j]))

# sort the list
logloss.sort()

# select the first 100
logloss_100 = copy.deepcopy(logloss[:100])
logloss_np = np.array(logloss_100)
logloss_mean = np.mean(logloss_np)
logloss_std = np.std(logloss_np)
print(len(logloss))
print(logloss_mean)
print(logloss_std)
