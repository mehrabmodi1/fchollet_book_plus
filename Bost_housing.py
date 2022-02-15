#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:02 2022

@author: mehrab
"""

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

def build_model():
    #architecture
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_data.shape[1],) ) )
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    
    #fit properties
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = 'mae')
    return model


#importing and re-organizing data
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

#z-scoring each feature
mean_vals = train_data.mean(axis=0)
train_data = train_data - mean_vals
sd_vals = train_data.std(axis = 0)
train_data = train_data/sd_vals

#normalizing test data to training data mean and sd
test_data = test_data - mean_vals 
test_data = test_data/sd_vals

#setting up k-fold cross-validation
k = 4       #number of data splits = number of separately trained models
num_val_samples = np.floor(len(train_data)/k)
num_epochs = 100
all_scores = []

for i in range k:
    print('processing fold #', i)
    #current validation block
    val_data = train_data[i*num_val_samples:(i + 1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i + 1)*num_val_samples]
    
    #current training blocks (all the rest of the training data)
    parital_train_data = np.concatenate(
            train_data[:i*num_val_samples], 
            train_data[(i+1)*num_val_samples:)],
            axis = 0)
    parital_train_targets = np.concatenate(
            train_targets[:i*num_val_samples], 
            train_targets[(i+1)*num_val_samples:)],
            axis = 0)