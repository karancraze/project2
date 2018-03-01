import idx2numpy

x_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
x_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

'''
for x in range(10):
    exec('tr_{} = []'.format(x))

for x in range(10):
    exec('te_{} = []'.format(x))

for idx, val in enumerate(y_train):
        exec('tr_{}.append(x_train[{}])'.format(val,idx))        

for idx, val in enumerate(y_test):
        exec('te_{}.append(x_test[{}])'.format(val,idx))
        
for x in range(10):    
    exec('te_{} = np.asarray(te_{})'.format(x,x))
    exec('tr_{} = np.asarray(tr_{})'.format(x,x))

tr_0 = tr_0.reshape(tr_0.shape[0], img_rows, img_cols, 1)
te_0 = te_0.reshape(te_0.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

tr_0 = tr_0.astype('float32')
te_0 = te_0.astype('float32')
tr_0 /= 255
te_0 /= 255

'''

img_rows, img_cols = 28, 28

num_classes = 10

# convert class vectors to binary class matrices
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255



import keras
from keras.utils import to_categorical

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size = 64
#acts = []

history = model.fit(x_train, y_train, 
                    epochs=5, 
                    batch_size=batch_size,
                    validation_split=0.2,
                    )


model.save('mnist_97.h5')
model.save_weights('mnist_97_weights.h5')


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()




'''
model2 = models.Sequential()
model2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

model2.layers[0].set_weights(model.layers[0].get_weights())
#model2.summary()

#building the dataset for training the other model which has activations as inputs
acts.append(model2.predict(x_train[0:1]))
'''