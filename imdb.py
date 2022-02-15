#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:22:13 2022

@author: mehrab
"""

from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    
    return results


#getting train, test data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
max_len = max([max(sequence) for sequence in train_data])       #computing lenght of longes movie review

#re-organizing data, labels into binary vectors
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#setting aside validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#specifying model architecture
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000,)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

#specifying fit type
model.compile(optimizer = 'rmsprop', 
              loss = losses.binary_crossentropy,
              metrics = [metrics.binary_accuracy])

#specifying fit parameters, inputs
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 4,
                    batch_size = 512,
                    validation_data = (x_test, y_test) )

#re-orgaizing fit metric data
history_dict = history.history
history_dict.keys()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

#plotting loss over training epochs
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#plotting accuracies over training epochs
plt.clf()
acc_vals = history_dict['binary_accuracy']
val_acc_vals = history_dict['val_binary_accuracy']

plt.plot(epochs, acc_vals, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_vals, 'b', label = 'Validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()







