#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:30:50 2022

@author: mehrab

This script computes the maximally activating image for a given unit in a given layer. 
It travels through image space for a given unit.
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K    #not sure why we aren't just using Numpy

model = VGG16(weights = 'imagenet', 
              include_top = False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output

#defining the loss as the mean output of the layer specified in layer_name
loss = K.mean(layer_output[:, :, :, filter_index])

#getting gradients of loss with small changes in input image
grads = K.gradients(loss, model.input)[0]      #only index 0 has a non-size1 tensor

#normalizing change made to fitted perfect image in each step by dividing gradients by L2 norm
grads = grads/(K.sqrt(K.mean(K.square(grads))) + 1e-5)      #adding 1e-5 in the end to prevent divide by 0 numerical error


#defining a function that takes an image as input and yeilds loss and gradients of loss
iterate = K.function([model.input], [loss, grads])
import numpy as np
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3) )])


#processing fitted activation image tensor to plot as an image
def deprocess_image(x):
    x = x - x.mean()
    x = x/(x.std() + 1e-5)
    x = x*0.1
    
    x = x + 0.5
    x = np.clip(x, 0, 1)    #clips z-scored image to [0, 1] before multiplying by 255
    x = x*255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


#main loop that fits the ideal activation image for current unit
input_img_data = np.random.random((1, 150, 150, 3) ) * 20 + 128.
step = 1.

for i in range(40):
    loss_value, grads_value = iterate([input_img_data])     #not sure how this works, K backend function def syntax is unusual
    input_img_data = input_img_data + grads_value * step
    step = step + 1
    