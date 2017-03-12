import os
import cv2
import scipy.io as sio
from shutil import copyfile


os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Ytrain = dataSet['Y_train']

images = []
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/train')
for i in range(Ytrain.shape[1]):
    copyfile(str(i + 1) + '.png', str(Ytrain[0, i]) + '/' + str(i + 1) + '.png')


