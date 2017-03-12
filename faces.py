import os
import numpy as np
import cv2
import scipy.io as sio



os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Ytrain = dataSet['Y_train'].flatten()

faces = []
bounds = []


faces = []
images = []
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/train')
for i in range(Ytrain.shape[0]):
    image = cv2.imread(str(i + 1) + '.png', 0)
    images.append(image)
os.chdir('/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades')
files = os.listdir('/usr/local/Cellar/opencv/2.4.13.2/share/OpenCV/haarcascades')
faceCascades = map(lambda f: cv2.CascadeClassifier(f), filter(lambda f: 'face' in f, files))

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/faces')
for k in range(len(images)):
    face = None
#    image = images[k]
    image = cv2.fastNlMeansDenoising(images[k])
    for fc in faceCascades:
        id = 0
        face = fc.detectMultiScale(image, [], [], scaleFactor = 1.001, minNeighbors = 6)
        if face != ():
            for x, y, w, h in face:
                cv2.imwrite('%d %d %d.png' % (k, faceCascades.index(fc), id), image[y: y + h, x: x + w])
                id += 1