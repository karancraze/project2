import idx2numpy
import pandas as pd
from keras import backend as K

x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
x_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')


img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

'''
zero_Images = []
one_Images = []
two_Images = []
three_Images = []
four_Images = []
five_Images = []
six_Images = []
seven_Images = []
eight_Images = []
nine_Images = []

for idx, val in enumerate(y_train):
    if val == 0:
        zero_Images.append(x_train[idx])
    if val == 1:
        one_Images.append(x_train[idx])
    if val == 2:
        two_Images.append(x_train[idx])
    if val == 3:
        three_Images.append(x_train[idx])
    if val == 4:
        four_Images.append(x_train[idx])
    if val == 5:
        five_Images.append(x_train[idx])
    if val == 6:
        six_Images.append(x_train[idx])
    if val == 7:
        seven_Images.append(x_train[idx])
    if val == 8:
        eight_Images.append(x_train[idx])
    if val == 9:
        nine_Images.append(x_train[idx])
  

zero_Images = zero_Images.reshape(zero_Images.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)
'''
import keras
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

batch_size = 64
acts = []


model.fit(x_train[0:1], y_train[0:1],
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_data=(x_test, y_test))

model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

model2.layers[0].set_weights(model.layers[0].get_weights())
#model2.summary()

#building the dataset for training the other model which has activations as inputs
acts.append(model2.predict(x_train[0:1]))
