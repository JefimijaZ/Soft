import numpy as np
import image as im
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


np.random.seed(7)
K.set_image_dim_ordering('th')
print K.image_data_format()
img_row, img_col = 28, 28
epochs, batch_size = 10, 200

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = im.prepare(x_train)
x_test = im.prepare(x_test)

x_train /= 255
x_test /= 255
x_train = x_train.reshape(x_train.shape[0], 1, img_row, img_col)
x_test = x_test.reshape(x_test.shape[0], 1, img_row, img_col)
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=epochs, batch_size=batch_size, verbose=2)
scores = model.evaluate(x_test, y_test, verbose=0)

model.save('model.h5')
