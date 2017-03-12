import os
import cv2
import scipy.io as sio
import numpy as np


os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Xtest = dataSet['X_test'].transpose()
Xtrain = dataSet['X_train'].transpose()
Ytrain = dataSet['Y_train'].transpose()


images = []
bounds = []

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/train')
for i in range(Ytrain.shape[0]):
    images.append(cv2.imread(str(i + 1) + '.png', 0))#cv2.IMREAD_GRAYSCALE))

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
transformedTrain = np.empty((Ytrain.shape[0], 8100))
for k in range(Ytrain.shape[0]):
    gray = np.float32(images[k])
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    index = dst > 0.009 * dst.max()
    #img = cv2.fastNlMeansDenoising(images[k])
    img = images[k]
    width = img.shape[1]
    height= img.shape[0]
    xmin = width
    xmax = 0
    ymin = height
    ymax = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if index[i][j]:
                xmin = min(j, xmin)
                xmax = max(j, xmax)
                ymin = min(i, ymin)
                ymax = max(i, ymax)
    b = (xmin, ymin, xmax, ymax)
    transformed = img[ymin:(ymax + 1), xmin:(xmax + 1)]
    cv2.imwrite(str(k) + ".png", transformed)
    transformed = cv2.resize(transformed, (90, 90), interpolation = cv2.INTER_CUBIC)
    transformedTrain[k] = transformed.flatten()

    bounds.append(b)

from sklearn import svm
lin_clf = svm.LinearSVC()
print lin_clf.fit(transformedTrain[:400, :], Ytrain[:400, :]).score(transformedTrain[400:, :], Ytrain[400:, :])

testImage = []
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/test')
for i in range(Xtest.shape[0]):
    testImage.append(cv2.imread(str(i + 1) + '.png', 0))#cv2.IMREAD_GRAYSCALE))

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_test')
transformedTest = np.empty((Xtest.shape[0], 8100))
theta = 0.009
for k in range(Xtest.shape[0]):
    gray = np.float32(testImage[k])
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    index = dst > theta * dst.max()
    img = cv2.fastNlMeansDenoising(testImage[k])
    width = img.shape[1]
    height= img.shape[0]
    xmin = width
    xmax = 0
    ymin = height
    ymax = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if index[i][j]:
                xmin = min(j, xmin)
                xmax = max(j, xmax)
                ymin = min(i, ymin)
                ymax = max(i, ymax)
    b = (xmin, ymin, xmax, ymax)
    transformed = img[ymin:(ymax + 1), xmin:(xmax + 1)]
    cv2.imwrite(str(k) + ".png", transformed)
    transformed = cv2.resize(transformed, (90, 90), interpolation = cv2.INTER_CUBIC)
    transformedTest[k] = transformed.flatten()



svm_result = lin_clf.predict(transformedTest)
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
np.savetxt("svm_result2.csv", np.dstack((np.arange(1, svm_result.size+1),svm_result))[0],"%d,%d",header="Id,ClassLabel", comments='')



'''
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (i == xmin or i == xmax) and ymin <= j <= ymax:
            img[i][j] = 255
        if (j == ymin or j == ymax) and xmin <= i <= xmax:
            img[i][j] = 255
'''
# = [0, 0, 255]

'''
cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''



