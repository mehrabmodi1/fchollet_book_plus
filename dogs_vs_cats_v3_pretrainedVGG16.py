#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 13:50:34 2022

@author: mehrab
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import numpy as np
import matplotlib.pyplot as plt

#specifying paths
base_dir = '/home/mehrab/ML_Datasets/DogsVsCats/subset/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#instantiating a model previously trained on the large, ImageNet dataset
conv_base = VGG16(weights = 'imagenet', 
                 include_top = False,           #using only the bottom (early) convnet layers and not using the top, dense, classifier layers 
                 input_shape = [150, 150, 3])



#reading in image data and saving output of pre-trained convnet to use as input to train a classifier later on
datagen = ImageDataGenerator(rescale = 1./255)      #specifying data generator object
batch_size = 20

#function to run un-augmented data through conv-net to generate extracted features to fit classifier later
def extract_features(directory, sample_count):
    features = np.zeros(shape = (sample_count, 4, 4, 512))
    labels = np.zeros(shape = (sample_count))
    generator = datagen.flow_from_directory(
            directory,
            target_size = (150, 150),
            batch_size = batch_size,
            class_mode = 'binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i*batch_size:(i+1)*batch_size, :, :, :] = features_batch
        labels[i*batch_size:(i+1)*batch_size] = labels_batch
        i = i + 1
        if i * batch_size >= sample_count:
            break
        return features, labels

#getting extracted features for the three datasets we have     
train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)


#flattening individual feature maps to vectors
train_features = np.reshape(train_features, (2000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

#specifying classifier layers that receive extracted convnet features as inputs
model = models.Sequential()
model.add(layers.Dense(256, activation = 'relu', input_dim = 4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = 'sigmoid'))

#specifying fit type
model.compile(optimizer = optimizers.RMSprop(lr = 2e-5),         #lr is learning rate
              loss = 'binary_crossentropy',
              metrics = 'acc')

#specifying fitting procedure and running fit for Dense layers of classifier
history = model.fit(train_features, train_labels,
                    epochs = 30,
                    batch_size = 20,
                    validation_data = (validation_features, validation_labels)
                    )

#plotting loss and accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'training accuracy')
plt.plot(epochs, val_acc, 'b', label = 'validation accuracy')
plt.title('training and validation accuracy')
plt.legend()

plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('training and validation loss')
plt.legend()
