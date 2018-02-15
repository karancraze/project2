'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam
import numpy as np


batch_size = 128
num_classes = 10
epochs = 30


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train=x_train[:100]
x_test=x_test[:100]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train=y_train[:100]
y_test=y_test[:100]

#i=0
acts = []

for i in range(0,10):
    #initial model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
              optimizer=adam(),
              metrics=['accuracy'])
    
    
    history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=1,
                    verbose=1,
                    validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    
    '''
    mediatary model for getting the activations
    gets weight from the main model
    doesn't have final layer
    '''
    model2 = Sequential()
    
    model2.add(Dense(64, activation='relu', input_shape=(784,)))
    model2.layers[0].set_weights(model.layers[0].get_weights())
    
    model2.add(Dense(64, activation='relu', input_shape=(784,)))
    
    model2.summary()
    
    #building the dataset for training the other model which has activations as inputs
    acts.append(model2.predict(x_train[0:1]))
    
    model3 = Sequential()
    model3.add(Dense(64, activation='relu', input_shape=(64,)))
    model3.add(Dense(64, activation='relu'))
    model3.add(Dense(num_classes, activation='softmax'))
    model3.summary()
    #acts2 = model2.predict(x_train[0:1])
    
    #acts[i].shape

    model3.compile(loss='categorical_crossentropy',
              optimizer=adam(),
              metrics=['accuracy'])
    
    history = model3.fit(acts[i], z3,
                    batch_size=1,
                    epochs=1,
                    verbose=1)
                    validation_data=(x_test, y_test))

y_train.shape
y_train[i].shape
z = y_train[i]
z2 = np.array(z)
z3 = z2.reshape(1,-1)

    #print (model.layers[0].weights)
    #print (model2.layers[0].weights)
    
    
    #
    #print (acts.shape)