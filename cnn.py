from __future__ import division, print_function, absolute_import
import os
import cv2
import scipy.io as sio
import numpy as np
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.conv import highway_conv_2d, max_pool_2d, conv_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
import pickle

# Load the data set
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')


Ytrain = dataSet['Y_train'].reshape([-1, 1])

Xtrain = []
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
for i in range(425):
    img = cv2.imread(str(i) + '.png', 0)
    Xtrain.append(np.array(img, dtype = float))#cv2.IMREAD_GRAYSCALE))

# Shuffle the data
X, Y = shuffle(Xtrain, Ytrain)
X = X.reshape([-1, 90, 90, 1])

Y = Y.reshape([-1, ])
# Make sure the data is normalized
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
#img_aug.add_random_blur(sigma_max=3.)

# Define our network architecture:
network = input_data(shape=[None, 90, 90, 1])

#                     data_preprocessing=img_prep,
#                     data_augmentation=img_aug)

network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 20, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=1, show_metric=True, run_id='convnet_highway_mnist')

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_test')
Xtest = np.empty([150, 90, 90])
for i in range(150):
    Xtest[i] = np.array(cv2.imread(str(i) + '.png', 0), dtype = float)

Xtest = Xtest.reshape([-1, 90, 90, 1])

cnn_result = model.predict(Xtest)

np.savetxt("cnn_result.csv", np.dstack((np.arange(1, cnn_result.size+1),cnn_result))[0],"%d,%d", \
           header="Id,ClassLabel", comments='')
