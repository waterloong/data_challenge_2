from __future__ import division, print_function, absolute_import
import os
import cv2
import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 20
epochs = 200

# input image dimensions
img_rows, img_cols = 90, 90

# Load the data set
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')


Ytrain = dataSet['Y_train']
Ytrain = np.concatenate((Ytrain, Ytrain), 0)

Xtrain = np.empty(shape=(850, 90, 90))
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
for i in range(850):
    img = cv2.imread(str(i) + '.png', 0)
    Xtrain[i] = np.array(img, dtype = float)

Xtest = np.empty(shape=(150, 90, 90))
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_test')
for i in range(150):
    img = cv2.imread(str(i) + '.png', 0)
    Xtest[i] = np.array(img, dtype = float)


if K.image_data_format() == 'channels_first':
    Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, img_rows, img_cols)
    Xtest = Xtest.reshape(Xtest.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    Xtrain = Xtrain.reshape(Xtrain.shape[0], img_rows, img_cols, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain /= 255
Xtest /= 255
print('x_train shape:', Xtrain.shape)
print(Xtrain.shape[0], 'train samples')
print(Xtest.shape[0], 'test samples')

# convert class vectors to binary class matrices
Ytrain = keras.utils.to_categorical(Ytrain - 1, num_classes)

model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epochs,
          verbose=1)

cnn_result = model.predict(Xtest)

score = model.evaluate(Xtrain, Ytrain, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2')

with open('cnn_result.csv', 'w') as resultFile:
    resultFile.write('Id,ClassLabel\n')
    for i in range(150):
        biggest = 0
        p = 0
        for j in range(20):
            s = cnn_result[i, j]
            if s > biggest:
                biggest = s
                p = j + 1
        resultFile.write('%d,%d\n' % (i + 1, p))
