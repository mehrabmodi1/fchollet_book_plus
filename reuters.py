#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 15:30:03 2022

@author: mehrab
"""

from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    
    return results

def to_one_hot(labels, dimensions=46):
    results = np.zeros((len(labels), dimensions))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


#getting and re-organizing data
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words = 10000)
word_index = reuters.get_word_index()                                                   #each word as a key, with it's number code
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])        #number code as keys with each word as the dic entry
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0] ])

#re-organizing data, labels into binary vectors
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#setting aside validation data
x_val = x_train[1:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[1:1000]
partial_y_train = one_hot_train_labels[1000:]


#specifying model architecture
model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu', input_shape = (10000, ) ))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(46, activation = 'softmax'))

#specifying model fit type
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

#specifying fit procedure and running fit
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs = 9,
                    batch_size = 512,
                    validation_data = (x_test, one_hot_test_labels) )


#re-orgaizing fit metric data
history_dict = history.history
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
acc_vals = history_dict['accuracy']
val_acc_vals = history_dict['val_accuracy']

plt.plot(epochs, acc_vals, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_vals, 'b', label = 'Validation accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


