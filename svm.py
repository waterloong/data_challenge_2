import os
import cv2
import scipy.io as sio
from sklearn import svm
import numpy as np


os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Xtest = dataSet['X_test'].transpose()
Xtrain = dataSet['X_train'].transpose()
Ytrain = dataSet['Y_train'].transpose()

images = []
bounds = []

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/train')
for i in range(Ytrain.shape[1]):
    images.append(cv2.imread(str(i + 1) + '.png', 0))#cv2.IMREAD_GRAYSCALE))

lin_clf = svm.LinearSVC()
print lin_clf.fit(Xtrain[:400, :], Ytrain[:400, :]).score(Xtrain[400:, :], Ytrain[400:, :])
svm_result = lin_clf.predict(Xtest)
np.savetxt("svm_result.csv", np.dstack((np.arange(1, svm_result.size+1),svm_result))[0],"%d,%d",header="Id,ClassLabel", comments='')
