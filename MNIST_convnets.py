from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#prepping train and test data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#specifying model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1) ))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.Flatten())     #converting 3D output to 1D for Dense, classifier layers
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))     #final classification ouput layer


#specifying model solver parameters
model.compile(optimizer = 'rmsprop', 
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

#specifying model training cycle parameters and running fit
model.fit(train_images, train_labels, epochs = 5, batch_size = 64)
model.evaluate(test_images, test_labels)