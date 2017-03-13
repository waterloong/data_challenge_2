import os
import cv2
import scipy.io as sio
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Xtest = dataSet['X_test'].transpose()
Xtrain = dataSet['X_train'].transpose()
Ytrain = dataSet['Y_train'].transpose()


images = []
bounds = []

def findTurningPoint(arr, size):
    bestStart = 0
    bestScore = 0
    for i in range(0, 20) + range(90, 109):
        score = abs(arr[i] - arr[i + 1])
        #print arr[i], arr[i + 1]
        if score > bestScore:
            bestScore = score
            bestStart = i
        print score
    if bestStart > 55:
        bestStart -= size
    return bestStart

def extractImage(img):
    bestScore = 0
    for i in range(20):
        for j in range(20):
            bestImg = img[i:(i + 90), j:(j + 90)]
            score = np.mean(img)
            if score > bestScore:
                bestScore = score
                bestImg = img
    return bestImg

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/train')
for i in range(Ytrain.shape[0]):
    img = cv2.imread(str(i) + '.png', 0)
    #img = cv2.fastNlMeansDenoising(img)
    images.append(img)

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
for i in range(Ytrain.shape[0]):
    img = images[i]
    varv = np.mean(img, 0)
    varh = np.mean(img, 1)
    size = 90
    ystart = findTurningPoint(varh, size)
    xstart = findTurningPoint(varv, size)
    cv2.imwrite(str(i) + '.png', img[ystart:(ystart + size), xstart:(xstart + size)])
    # cv2.imwrite(str(i) + '.png', extractImage(img))


# for i in range(0, 10):
#     img = images[i]
#     varv = np.max(img, 0)
#     varh = np.max(img, 1)
#     plt.plot(varh)
# plt.show()