"""
It contains the new datasets that will be experimented with EvoDNN
"""

import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = "datasets/"

def load_abalone():
    """
    Load the abalone dataset from UCI ML repository
    """
    fp = ROOT + "abalone.csv"
    print("Loading abalone...")
    raw_data = pd.read_csv(fp, delimiter=',', header=None)
    x = np.array(raw_data.iloc[:,0])
    x2 = np.array(raw_data.iloc[:,1:8])
    x = LabelEncoder().fit_transform(x)
    data = np.c_[x, x2]
    target = np.array(raw_data.iloc[:,8])
    return data, target

def load_ecoli():
    fp = ROOT + 'ecoli.csv'
    print("Loading ecoli...")
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:, 1:8])
    y = np.array(raw_data.iloc[:, 8])
    y = LabelEncoder().fit_transform(y)
    return x, y

def load_haberman():
    fp = ROOT + 'haberman.csv'
    print("Loading haberman...")
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, :3])
    y = np.array(raw_data.iloc[:, 3])
    return x, y

def load_ilpd():
    fp = ROOT + 'ilpd.csv'
    print("Loading ilpd...")
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 0])
    x1 = np.array(raw_data.iloc[:, 1])
    x2 = np.array(raw_data.iloc[:, 2:10])
    x1 = LabelEncoder().fit_transform(x1)
    x = np.c_[x, x1, x2]
    y = np.array(raw_data.iloc[:, 10])
    return x, y

def load_lymphography():
    fp = ROOT + 'lymphography.csv'
    print("Loading lymphography...")
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:])
    y = np.array(raw_data.iloc[:, 0])
    return x, y

def load_mammographic():
    fp = ROOT + 'mammographic_masses.csv'
    print("Loading mammographic...")
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:5])
    y = np.array(raw_data.iloc[:, 5])
    return x, y

def load_spect():
    fp_test = ROOT + "spect_test.csv"
    fp_train = ROOT + "spect_train.csv"
    print("Loading spect...")
    raw_data_test = pd.read_csv(fp_test, header=None)
    raw_data_train = pd.read_csv(fp_train, header=None)
    frames = [raw_data_train, raw_data_test]
    raw_data = pd.concat(frames)
    x = np.array(raw_data.iloc[:,1:])
    y = np.array(raw_data.iloc[:, 0])
    return x, y

def load_spectf():
    fp_test = ROOT + "spectf_test.csv"
    fp_train = ROOT + "spectf_train.csv"
    print("Loading spectf...")
    raw_data_test = pd.read_csv(fp_test, header=None)
    raw_data_train = pd.read_csv(fp_train, header=None)
    frames = [raw_data_train, raw_data_test]
    raw_data = pd.concat(frames)
    x = np.array(raw_data.iloc[:,1:])
    y = np.array(raw_data.iloc[:, 0])
    return x, y

def load_statlog():
    fp = ROOT + "heart.txt"
    print("Loading statlog...")
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:,0:13])
    y = np.array(raw_data.iloc[:, 13])
    return x, y

def load_thoracic():
    fp = ROOT + "thoracic.csv"
    print("Loading thoracic...")
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:,0:16])
    y = np.array(raw_data.iloc[:, 16])
    return x, y

def load_tumor():
    fp = ROOT + "tumor.csv"
    print("Loading tumor...")
    raw_data = pd.read_csv(fp, header=None)
    x = np.array(raw_data.iloc[:, 1:3])
    x1 = np.array(raw_data.iloc[:, 5:18])
    y = np.array(raw_data.iloc[:, 0])
    x = np.c_[x, x1]
    return x, y

def load_wilt():
    fp_train = ROOT + "wilt_training.csv"
    fp_test = ROOT + "wilt_testing.csv"
    print("Loading wilt...")
    raw_data_train = pd.read_csv(fp_train, delimiter=',')
    raw_data_test = pd.read_csv(fp_test, delimiter=',')
    frames = [raw_data_train, raw_data_test]
    raw_data = pd.concat(frames)
    data = np.array(raw_data.iloc[:,1:6])
    y = np.array(raw_data.iloc[:,0])
    target = LabelEncoder().fit_transform(y)
    return data, target

def load_yeast():
    fp = ROOT + "yeast.data.txt"
    print("Loading yeast...")
    raw_data = pd.read_csv(fp, delim_whitespace=True, header=None)
    x = np.array(raw_data.iloc[:,1:5])
    x1 = np.array(raw_data.iloc[:,7:9])
    data = np.c_[x, x1]
    y = np.array(raw_data.iloc[:,9])
    target = LabelEncoder().fit_transform(y)
    return data, target

if __name__ == '__main__':
    load_abalone()
    load_ecoli()
    load_haberman()
    load_ilpd()
    load_lymphography()
    load_mammographic()
    load_spect()
    load_spectf()
    load_statlog()
    load_thoracic()
    load_tumor()
    load_wilt()
    load_yeast()
