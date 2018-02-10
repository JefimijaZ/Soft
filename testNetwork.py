from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import load_model
from keras import backend as K
num_classes = 10

# the data, shuffled and split between train and test sets
img_rows, img_cols = 28, 28
(x_ignore, y_ignore), (x_test, y_test) = mnist.load_data()
x_test = x_test[0:1048]
y_test = y_test[0:1048]

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
y_test = keras.utils.to_categorical(y_test, num_classes)
model = load_model('model.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])