#import module
import cv2
import numpy as np
import matplotlib.pyplot as plt

#path directory
path= "/content/drive/MyDrive/Assignment Python/"

#set image size
plt.figure(figsize=[20,20])

#open image
img = cv2.imread(path+ "coins.jpg")
plt.subplot(161);plt.imshow(img, cmap='gray');plt.title("Original Image")

#gray image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(162);plt.imshow(grayImg, cmap='gray');plt.title("Gray Image")

#blur image
blurImg = cv2.GaussianBlur(grayImg, (11,11), 0)
plt.subplot(163);plt.imshow(blurImg, cmap='gray');plt.title("Blur Image")

#canny edge detection image
cannyImg = cv2.Canny(blurImg, 30, 150, 3)
plt.subplot(164);plt.imshow(cannyImg, cmap='gray');plt.title("Edge Filtering Image")

#dilated image
dilatedImg = cv2.dilate(cannyImg, (1,1), iterations = 2)
plt.subplot(165);plt.imshow(dilatedImg, cmap='gray');plt.title("Dilated Image")

#count contours
(cnt, heirarchy) = cv2.findContours(dilatedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgbImg, cnt, -1, (0,0,255), 2)
print('Coins in the image: ', len(cnt))
plt.subplot(166);plt.imshow(rgbImg);plt.title("Coins Image")
plt.show()
