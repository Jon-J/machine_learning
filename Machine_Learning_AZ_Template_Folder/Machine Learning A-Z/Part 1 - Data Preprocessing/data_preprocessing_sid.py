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

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencode_X = LabelEncoder()
X[:, 0] = labelencode_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencode_Y = LabelEncoder()
Y = labelencode_Y.fit_transform(Y)
