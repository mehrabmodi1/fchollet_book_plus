import os
from os.path import exists
from keras import layers
from keras import models
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
import matplotlib.pyplot as plt

#reading in and pre-processing databases of images
base_dir = '/home/mehrab/ML_Datasets/DogsVsCats/subset/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

#setting up automatic image importing Python generator objects with image augmentation arguments
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   rotation_range = 40,     #randomly rotates images by +/- 40 degrees
                                   width_shift_range = 0.2,  #randomly shifts image along x by upto 0.2x width
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,        #distorts image by shifting each row by incremental amounts
                                   zoom_range = 0.2, 
                                   horizontal_flip = True,
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255) #augmentation here would be wrong - would create redundant test/val data and distort performance

train_generator = train_datagen.flow_from_directory(
        train_dir,                  #train_dir contains two sub-folders with the different classes. generator draws from each one randomly
        target_size = (150, 150),   #generator resizes all images to this xy size
        batch_size = 20,            #the number of images loaded in each iteration of the generator
        class_mode = 'binary')      #class mode specifies that labels should be binary

validation_generator = test_datagen.flow_from_directory(validation_dir, target_size = (150, 150), batch_size = 20, class_mode = 'binary')

#using generator objects to read in image batches in an example loop for one iteration
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break


#specifying model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

#specifying fit parameter types
model.compile(loss = 'binary_crossentropy', optimizer = optimizers.RMSprop(lr = 1e-4), metrics = ['acc'])

#specifying fit procedure parameters and fitting model
history = model.fit_generator(
        train_generator,        #source of each batch of data
        steps_per_epoch = 100,  #specifying how many batches to draw per epoch (2000 images in training set) since generator loops endlessly
        epochs = 100,
        validation_data = validation_generator,
        validation_steps = 50  #specifying how many batches of validation images to draw from looping generator (1000 val images)
        )

save_path = '/home/mehrab/ML_Datasets/saved_models/cats_and_dogs_small_2data_aug.h5'
if exists(save_path) == 1:
    save_path = os.path.join(save_path[0:-3], '1.h5')       #making sure to not accidentally over-write a previous model
model.save(save_path)

#plotting training, validation loss and accuracy over training epochs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, (len(acc) + 1))
plt.plot(epochs, acc, 'bo', label = 'training acc')
plt.plot(epochs, val_acc, 'b', label = 'validation acc')
plt.title('training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label = 'training loss')
plt.plot(epochs, val_loss, 'b', label = 'validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()




 

