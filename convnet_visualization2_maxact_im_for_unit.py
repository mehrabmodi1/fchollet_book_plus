#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:30:50 2022

@author: mehrab

This script computes the maximally activating image for a given unit in a given layer. 
It travels through image space for a given unit.

Note that re-running the fit doesn't generate an identical best-activation image for but an image with very similar feature statistics!!
This is not a simple offset, which is most noticeable in block 5 units
"""

#Use big, pre-trained network
from tensorflow.keras.applications import VGG16
model = VGG16(weights = 'imagenet', include_top = False)

#use home-trained dog/cat classifier  
#from tensorflow.keras.models import load_model
#model = load_model('/home/mehrab/ML_Datasets/saved_models/cats_and_dogs_small_2data_aug.h5')


from tensorflow.keras import backend as K    #not sure why we aren't just using Numpy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()      #gradients is deprecated in TF2

#manual input variables
layer_name = 'block5_conv1'
filter_index = 3


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


#main function that fits the ideal image to activate the nth unit in the current layer
def generate_pattern(layer_name, filter_index, size = 150):
    layer_output = model.get_layer(layer_name).output

    #defining the loss as the mean output of the layer specified in layer_name
    loss = K.mean(layer_output[:, :, :, filter_index])
    #getting gradients of loss with small changes in input image
    grads = K.gradients(loss, model.input)[0]      #only index 0 has a non-size1 tensor
    #normalizing change made to fitted perfect image in each step by dividing gradients by L2 norm
    grads = grads/(K.sqrt(K.mean(K.square(grads))) + 1e-5)      #adding 1e-5 in the end to prevent divide by 0 numerical error
    
    #defining a function 'iterate' acc to Keras.backend convention, with specified outputs and inputs, weird syntax
    iterate = K.function([model.input], [loss, grads])      
   
    #initializing a middle-of-range image with some random noise
    input_img_data = np.random.random((1, size, size, 3) ) * 20 + 128.
    
    step = 1.
    
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])     #not sure how this works, K.backend function def syntax is unusual
        #pdb.set_trace()
        input_img_data = input_img_data + grads_value * step
        step = step + 1     #added this line because otherwise added gradient is decreasing proportionately on every iteration
    
    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern(layer_name, filter_index))
