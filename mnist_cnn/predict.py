from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


model = load_model('mnist_97.h5')
model.load_weights('mnist_97_weights.h5')

model.summary()

def load_image(img_path, show=True):
    img = image.load_img(img_path, target_size=(28, 28))
    plt.imshow(img)                           
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    img = np.reshape(img,[1,28,28,1])
    return img


img_path = '3.png'
img_path2 = '2.png'
img_path3 = '7.png'

img = load_image(img_path)
img = load_image(img_path2)
img = load_image(img_path3)

pred = []
pred.append(model.predict(img))
