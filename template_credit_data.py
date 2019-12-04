#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:30:12 2019

@author: martoso
"""

# Import modules
import collections
import pandas as pd
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Import dataset
df = pd.read_csv('./credit-data.csv')
df.loc[df.age < 0, 'age'] = df.loc[df.age > 0].mean()
df.describe()

# Split dataset in Forecasters and Classes
forecasters = df.iloc[:, 1:4].values
classes = df.iloc[:, 4].values

# Fix missing values
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(forecasters[:, 1:4])
forecasters[:, 1:4] = imputer.transform(forecasters[:, 1:4])

# Scaling values
scaler = StandardScaler()
forecasters = scaler.fit_transform(forecasters)

# Split dataset into Train and Test data
forecasters_train, forecasters_test, classes_train, classes_test = train_test_split(
    forecasters,
    classes,
    test_size=0.25,
    random_state=0)

# Machine Learning import
# Classifier creation
estimator = ()
# estimator.fit(forecasters_train, classes_train)

# Predict new Classes
predictions = 0
# predictions = estimator.predict(forecasters_test)

# Predict Precision
precision = accuracy_score(classes_test, predictions)
print('Precision: {:.2f}%'.format(precision * 100))

# Confusion Matrix
matrix = confusion_matrix(classes_test, predictions)
print('Confusion Matrix: \n\t0\t1\n0:\t{}\t{} \n1:\t{}\t{}'
      .format(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))

# Line Base Classifier
counter = collections.Counter(classes_test)
print('Class A: {}\nClass B: {}\n\n\tLine Base Classifier: {}%'
      .format(counter[0], counter[1], counter[0]/(counter[0]+counter[1])*100))
