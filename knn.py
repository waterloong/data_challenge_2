import numpy as np
import os
import scipy.io as sio
import cv2
from sklearn.neighbors import KNeighborsClassifier

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Ytrain = dataSet['Y_train']
Ytrain = np.concatenate((Ytrain, Ytrain), 1)
Ytrain = Ytrain.transpose().astype(int)


transformedTrain = np.empty((850, 8100))
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
for i in range(850):
    img = cv2.resize(cv2.imread(str(i) + '.png', 0), (90, 90), interpolation = cv2.INTER_CUBIC)
    transformedTrain[i] = img.flatten()


nbrs = KNeighborsClassifier(n_neighbors=1)
#print nbrs.fit(transformedTrain[:400, :], Ytrain[:400, :]).score(transformedTrain[400:, :], Ytrain[400:, :])
nbrs.fit(transformedTrain, Ytrain)

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_test')
transformedTest = np.empty((150, 8100))
for i in range(150):
    img = cv2.resize(cv2.imread(str(i) + '.png', 0), (90, 90), interpolation = cv2.INTER_CUBIC)
    transformedTest[i] = img.flatten()
knn_result = nbrs.predict(transformedTest)

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2')
np.savetxt("knn_result3.csv", np.dstack((np.arange(1, knn_result.size+1),knn_result))[0],"%d,%d",header="Id,ClassLabel", comments='')





