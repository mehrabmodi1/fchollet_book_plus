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
import matplotlib.pyplot as plt

def build_model():
    #architecture
    model = models.Sequential()
    model.add(layers.Dense(32, activation = 'relu', input_shape = (train_data.shape[1],) ) )
    #model.add(layers.Dense(32, activation = 'relu'))
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


do_validation = 0

if do_validation == 1:
    #setting up k-fold cross-validation
    k = 4       #number of data splits = number of models separately trained on diff combos of splits
    num_val_samples = len(train_data)//k
    n_epochs = 80
    all_scores = []
    all_mae_histories = []

    #running fit, validating on different combinations of data blocks
    for i in range(k):
        print('processing fold #', i)
        #current validation block
        val_data = train_data[i*num_val_samples:(i + 1)*num_val_samples]
        val_targets = train_targets[i*num_val_samples:(i + 1)*num_val_samples]
        
        #current training blocks (all the rest of the training data)
        partial_train_data = np.concatenate(
                (train_data[:i*num_val_samples], 
                train_data[(i+1)*num_val_samples:]),
                axis = 0)
        partial_train_targets = np.concatenate(
                (train_targets[:i*num_val_samples], 
                train_targets[(i+1)*num_val_samples:]),
                axis = 0)
        
        #setting up and fitting model for current block of training data
        model = build_model()
        history = model.fit(partial_train_data, partial_train_targets,
                            validation_data = (val_data, val_targets),
                  epochs = n_epochs, batch_size = 1, verbose = 0)
        #keeping track of performance on val block over training epochs
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)
            
        #evaluating current block of cross-val data on currently fitted model
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose = 0)
        all_scores.append(val_mae)
        print('done')
        
        #plotting validation block performace over training epochs
        ave_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(n_epochs)]
        
        def smooth_curve(points, factor = 0.9):
            smoothed_points = []
            for point in points:
                if smoothed_points:
                    previous = smoothed_points[-1]
                    smoothed_points.append(previous * factor + point * (1 - factor))
                else:
                    smoothed_points.append(point)
            return smoothed_points
                        
        smooth_mae_history = smooth_curve(ave_mae_history[10:])
        plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
        plt.xlabel('Epochs')
        plt.ylabel('smth. Validation MAE')
        plt.show()
        
elif do_validation == 0:
    model = build_model()
    #training on entire training set now
    model.fit(train_data, train_targets,
              epochs = 80, batch_size = 16, verbose = 0)
    #evaluating test data
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)






    