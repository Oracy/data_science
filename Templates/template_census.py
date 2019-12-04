#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:25:51 2019

@author: martoso
"""

# Import modules
import collections
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Import dataset
df = pd.read_csv('./census.csv')
df.describe()

# Split dataset in Forecasters and Classes
forecasters = df.iloc[:, 0:14].values
classes = df.iloc[:, 14].values

# Encode categorical variables
label_encoder_forecasters = LabelEncoder()
forecasters[:, 1] = label_encoder_forecasters.fit_transform(forecasters[:, 1])
forecasters[:, 3] = label_encoder_forecasters.fit_transform(forecasters[:, 3])
forecasters[:, 5] = label_encoder_forecasters.fit_transform(forecasters[:, 5])
forecasters[:, 6] = label_encoder_forecasters.fit_transform(forecasters[:, 6])
forecasters[:, 7] = label_encoder_forecasters.fit_transform(forecasters[:, 7])
forecasters[:, 8] = label_encoder_forecasters.fit_transform(forecasters[:, 8])
forecasters[:, 9] = label_encoder_forecasters.fit_transform(forecasters[:, 9])
forecasters[:, 13] = label_encoder_forecasters.fit_transform(forecasters[:, 13])

# Encode categorical to column values
one_hot_encoder = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
forecasters = one_hot_encoder.fit_transform(forecasters).toarray()

# Encode Classes to numeric
label_encoder_classes = LabelEncoder()
classes = label_encoder_classes.fit_transform(classes)

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

# Line Base Classifier
counter = collections.Counter(classes_test)
print('Class A: {}\nClass B: {}\n\n\tLine Base Classifier: {}%'
      .format(counter[0], counter[1], counter[0]/(counter[0]+counter[1])*100))

# Predict Precision
precision = accuracy_score(classes_test, predictions)
print('Precision: {:.2f}%'.format(precision * 100))

# Confusion Matrix
matrix = confusion_matrix(classes_test, predictions)
print('Confusion Matrix: \n\t0\t1\n0:\t{}\t{} \n1:\t{}\t{}'
      .format(matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]))
