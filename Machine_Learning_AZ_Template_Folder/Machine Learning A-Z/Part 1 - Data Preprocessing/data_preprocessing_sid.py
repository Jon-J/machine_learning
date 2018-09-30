# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:32:31 2018

@author: sid
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
datasets = pd.read_csv('Data.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])