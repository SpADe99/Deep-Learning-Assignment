#import module
import cv2
import numpy as np
import matplotlib.pyplot as plt

path= "/content/drive/MyDrive/Assignment Python/"

#open image
img = cv2.imread(path+ "coins.jpg")

#gray image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(grayImg, cmap='gray');

#blur image
blurImg = cv2.GaussianBlur(grayImg, (11,11), 0)
plt.imshow(blurImg, cmap='gray')

#canny edge detection image
cannyImg = cv2.Canny(blurImg, 30, 150, 3)
plt.imshow(cannyImg, cmap='gray')

#dilated image
dilatedImg = cv2.dilate(cannyImg, (1,1), iterations = 2)
plt.imshow(dilatedImg, cmap='gray')

#count contours
(cnt, heirarchy) = cv2.findContours(dilatedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgbImg, cnt, -1, (0,0,255), 2)
plt.imshow(rgbImg)
print('Coins in the image: ', len(cnt))
plt.show()
