#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 17:51:05 2022

@author: mehrab
This script visualises a single image convolved with the feature fitted to each unit of each layer and sequentially down-sampled. 
It travels through unit space for a given image.

Early layers should be simpler features. The model used here was entirely trained locally with 2000 base images augmented by transformations.

#Notes:
- Can't make out dog v/s cat from last layer's outputs by human eye. Very few blank filters even in last layer as compared to text book image.
- Didn't really find 'ear detectors'. The same unit was activated very differently by different images of cats or cat v/s dog.
"""

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt


model = load_model('/home/mehrab/ML_Datasets/saved_models/cats_and_dogs_small_2data_aug.h5')

#img_path = '/home/mehrab/ML_Datasets/DogsVsCats/subset/test/cats/cat.1701.jpg'
img_path = '/home/mehrab/ML_Datasets/DogsVsCats/subset/test/dogs/dog.1700.jpg'
img = image.load_img(img_path, target_size = (150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis = 0)
img_tensor = img_tensor/255

plt.imshow(img_tensor[0])
plt.show()
plt.figure()

layer_outputs = [layer.output for layer in model.layers[:8]]        #this extracts the outputs of the first 8 layers
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)      #creates a model that will return specified outpus, given the input

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 1], cmap = 'viridis')

#visualizing activation of each feature by current input image
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features//images_per_row
    
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :, 
                                             (col*images_per_row) + row]
            channel_image = channel_image - channel_image.mean()
            channel_image = channel_image/channel_image.std()
            channel_image = (channel_image*64) + 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size:(col+1)*size,
                         row*size:(row+1)*size] = channel_image
            
    
    scale = 1. /size
    plt.figure(figsize = (scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')