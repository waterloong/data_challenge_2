import os
import cv2
import scipy.io as sio
import numpy as np


os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

#Ytrain = dataSet['Y_train'].transpose()
m = 425

images = []
kpTrain = []

sift = cv2.SIFT()
bf = cv2.BFMatcher()

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/train')
for i in range(m):
    img = cv2.imread(str(i + 1) + '.png', 0)
    kp, des = sift.detectAndCompute(img, None)
    kpTrain.append((kp, des))
#    images.append(img)

for i in range(400, 425):
    kp1, des1 = kpTrain[i]
    best = (0, 0)
    for j in range(400):
        kp2, des2 = kpTrain[j]
        matches = bf.knnMatch(des1, des2, k = 2)
        count = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                pass


