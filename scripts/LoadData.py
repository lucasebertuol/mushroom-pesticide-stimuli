from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import numpy as np

class LoadData:

    #------------------------------------------
    def __init__(self):
        pass
    
    #------------------------------------------
    def load_classes(self, classes_file):

        classes = np.loadtxt(classes_file)
        return classes
    
    #------------------------------------------
    def load_signals(self, signals_file):

        signals = np.loadtxt(signals_file, delimiter=';')
        return signals

    #------------------------------------------
    def split_data(self, x, y):
   
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        return x_train, x_test, y_train, y_test

    #------------------------------------------
    def scaler(self, data):

        s = StandardScaler()
        scaled = s.fit_transform(data)
        return scaled

    #------------------------------------------
    
