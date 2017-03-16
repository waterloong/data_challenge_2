import cv2
import os
from PIL import Image
import numpy as np

os.chdir('/Users/William/Google Drive/UW/STAT441/data_challenge_2/transformed_train')
for i in range(425):
    # img = cv2.imread(str(i) + '.png', 0)
    # cv2.imwrite(str(i + 425) + '.png', np.fliplr(img))
    img = Image.open(str(i) + '.png')
    img2 = img.rotate(10)
    img2.save(str(i + 425 * 2) + '.png')
    img2 = img.rotate(-10)
    img2.save(str(i + 425 * 3) + '.png')
    img2 = img.rotate(20)
    img2.save(str(i + 425 * 4) + '.png')
    img2 = img.rotate(-20)
    img2.save(str(i + 425 * 5) + '.png')

