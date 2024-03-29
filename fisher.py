import os
import numpy as np
import cv2
import scipy.io as sio

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/')
dataSet = sio.loadmat('FacesDataChallenge.mat')

Ytrain = dataSet['Y_train']
Ytrain = np.concatenate((Ytrain, Ytrain, Ytrain), 1)
Ytrain = np.concatenate((Ytrain, Ytrain), 1)

Ytrain = Ytrain.transpose().astype(int)
recognizer = cv2.createEigenFaceRecognizer()
#recognizer = cv2.createFisherFaceRecognizer()
#recognizer = cv2.createLBPHFaceRecognizer()

images = []

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
for i in range(425 * 6):
    image = cv2.imread(str(i) + '.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image = cv2.resize(image, (90, 90), interpolation = cv2.INTER_CUBIC)
    images.append(image)

recognizer.train(images, Ytrain)

testImages = []
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_test')
for i in range(150):
    image = cv2.imread(str(i) + '.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    image = cv2.resize(image, (90, 90), interpolation = cv2.INTER_CUBIC)
    testImages.append(image)
os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2')

with open('eigen_result3.csv', 'w') as resultFile:
    resultFile.write('Id,ClassLabel\n')
    for i in range(150):
        result, conf = recognizer.predict(testImages[i])
        resultFile.write('%d,%d\n' % (i + 1, result))
        print i, conf
