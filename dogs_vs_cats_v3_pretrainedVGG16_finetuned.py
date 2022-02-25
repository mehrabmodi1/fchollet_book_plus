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

#specifying classifier layers that receive extracted convnet features as inputs
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu', input_dim = 4*4*512))
model.add(layers.Dense(1, activation = 'sigmoid'))

#freezing weights of the pre-trained convnet
conv_base.trainable = False

#setting up augmented data generation for training and regular data read-in for validation
train_datagen = ImageDataGenerator(rescale = 1./255,
                             rotation_range = 40,
                             width_shift_range = 0.2,
                             height_shift_range = 0.2,
                             shear_range = 0.2,
                             zoom_range = 0.2,
                             horizontal_flip = True,
                             fill_mode = 'nearest')      #specifying data generator object

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size = (150, 150),
                                                    batch_size = 20,
                                                    class_mode = 'binary'
                                                    )

test_datagen = ImageDataGenerator(rescale = 1./255)     #no data augmentation for test or validation data

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size = (150, 150),
                                                        batch_size = 20,
                                                        class_mode = 'binary'
                                                        )

#specifying fit type
model.compile(optimizer = optimizers.RMSprop(lr = 2e-5),         #lr is learning rate
              loss = 'binary_crossentropy',
              metrics = 'acc')

#specifying fitting procedure and running fit for Dense layers of classifier
history = model.fit_generator(train_generator,
                    steps_per_epoch = 100,                     
                    epochs = 30,
                    validation_data = (validation_generator),
                    validation_steps = 50
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

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('training and validation loss')
plt.legend()
