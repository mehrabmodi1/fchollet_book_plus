#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 14:33:28 2022

@author: mehrab

Note: Gave a challenging image with two elephants with intertwined trunks and the weak prediction was Camel
"""

from tensorflow.keras.applications import VGG16
model = VGG16(weights = 'imagenet')

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K  
import numpy as np
import cv2

img_path = '/home/mehrab/Downloads/elephant_challenge.jpg'              #easy to find online. ought to work with other images
img = image.load_img(img_path, target_size = (224, 224))
x = image.img_to_array(img)         
x = np.expand_dims(x, axis = 0)     #adding an extra dim at 0 for batch
x = preprocess_input(x)     #probably z-scoring etc

preds = model.predict(x)
class_n = np.argmax(preds[0])     #identifying class number with max predicted probability
print('predicted: ', decode_predictions(preds, top = 3)[0])    #printing predicted class-labels for top 3

class_output = model.output[:, class_n]                 #entry for class_n in prediction vector
last_conv_layer = model.get_layer('block5_conv3')       #output of final convnet layer

#generating class activation map
grads = K.gradients(class_output, last_conv_layer.output)[0]    #gradient of class pred prob with small changes in output feature map of last layer.
pooled_grads = K.mean(grads, axis = (0, 1, 2) )         #vector of shape 512 (ie. n units). each entry is the 'mean intensity of the gradient over a specific feature map channel'

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] = conv_layer_output_value[:, :, i] * pooled_grads_value[i]     #layer output is from 512 units of output size 14x14, multiplying each unit's output by it's importance for class
    
heatmap = np.mean(conv_layer_output_value, axis = 2)    #averaging across units after they have been weighted
#heatmap = heatmap / np.max(heatmap)
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0] ))    #heatmap is just 14x14
heatmap = heatmap + np.abs(np.min(heatmap))     #adding offset to make everything positive
heatmap = heatmap / np.max(heatmap)        #notmalising
heatmap = np.uint8(255*heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('/home/mehrab/Downloads/superimposed_img_CAMmap.jpg', superimposed_img)
