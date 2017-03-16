import os
import cv2
import scipy.io as sio
import numpy as np


os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Xtest = dataSet['X_test'].transpose()
Xtrain = dataSet['X_train'].transpose()
Ytrain = dataSet['Y_train']
Ytrain = dataSet['Y_train']
Ytrain = np.concatenate((Ytrain, Ytrain, Ytrain), 1)
Ytrain = np.concatenate((Ytrain, Ytrain), 1)
Ytrain = Ytrain.transpose()


bounds = []

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
transformedTrain = np.empty((Ytrain.shape[0], 8100))
for k in range(Ytrain.shape[0]):
    transformed = cv2.imread(str(k) + '.png', 0)
    transformedTrain[k] = transformed.flatten()

from sklearn import svm
lin_clf = svm.LinearSVC()
#print lin_clf.fit(transformedTrain[:400, :], Ytrain[:400, :]).score(transformedTrain[400:, :], Ytrain[400:, :])
print lin_clf.fit(transformedTrain, Ytrain)

testImage = []
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/test')
for i in range(Xtest.shape[0]):
    testImage.append(cv2.imread(str(i + 1) + '.png', 0))

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_test')
transformedTest = np.empty((Xtest.shape[0], 8100))
for k in range(Xtest.shape[0]):
    transformed = cv2.imread(str(k) + ".png", 0)
    transformedTest[k] = transformed.flatten()

svm_result = lin_clf.predict(transformedTest)
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
np.savetxt("svm_result6.csv", np.dstack((np.arange(1, svm_result.size+1),svm_result))[0],"%d,%d",header="Id,ClassLabel", comments='')


'''
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''



