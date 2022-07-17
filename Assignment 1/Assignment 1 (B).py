#import module
import cv2
import numpy as np
import matplotlib.pyplot as plt

#path directory
path= "/content/drive/MyDrive/Assignment Python/"

#open image
happyImg = cv2.imread(path+ "happy.jpg")
sadImg = cv2.imread(path+ "sad.jpg")
shockImg = cv2.imread(path+ "shock.jpg")

#convert image to RGB
happyImg = cv2.cvtColor(happyImg, cv2.COLOR_BGR2RGB)
sadImg = cv2.cvtColor(sadImg, cv2.COLOR_BGR2RGB)
shockImg = cv2.cvtColor(shockImg, cv2.COLOR_BGR2RGB)

#image for blur
happyImgBlur = np.copy(happyImg)
sadImgBlur = np.copy(sadImg)
shockImgBlur = np.copy(shockImg)

#initialize the Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

#face bounding box coordinates using Haar Cascade
detections1 = face_cascade.detectMultiScale(happyImg)
detections2 = face_cascade.detectMultiScale(sadImg)
detections3 = face_cascade.detectMultiScale(shockImg)

#blurr image 1
for face in detections1:
  x,y,w,h = face
  #blur happy image
  happyImgBlur[y:y+h,x:x+w] = cv2.GaussianBlur(happyImgBlur[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
  cv2.rectangle(happyImgBlur,(x,y),(x+w,y+h),(255,0,0),2)

#blurr image 2
for face in detections2:
  x,y,w,h = face
  #blur sad image
  sadImgBlur[y:y+h,x:x+w] = cv2.GaussianBlur(sadImgBlur[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
  cv2.rectangle(sadImgBlur,(x,y),(x+w,y+h),(255,0,0),2)

#blurr image 3
for face in detections3:
  x,y,w,h = face
  #blur shock image 
  shockImgBlur[y:y+h,x:x+w] = cv2.GaussianBlur(shockImgBlur[y:y+h,x:x+w],(15,15),cv2.BORDER_DEFAULT)
  cv2.rectangle(shockImgBlur,(x,y),(x+w,y+h),(255,0,0),2)

#display image
plt.figure(figsize=[7,11])
plt.subplot(321);plt.imshow(happyImg);plt.title("Happy Image");
plt.subplot(322);plt.imshow(happyImgBlur);plt.title("Blur Happy Image");
plt.subplot(323);plt.imshow(sadImg);plt.title("Sad Image");
plt.subplot(324);plt.imshow(sadImgBlur);plt.title("Blur Sad Image");
plt.subplot(325);plt.imshow(shockImg);plt.title("Shock Image");
plt.subplot(326);plt.imshow(shockImgBlur);plt.title("Blur Shock Image");
