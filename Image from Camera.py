# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:56:41 2021

@author: S Rani
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

videoCaptureObject = cv2.VideoCapture(0)
result=True
while(result):
    ret,frame=videoCaptureObject.read()
    cv2.imwrite("test_for_camera.jpg",frame)
    result=False
videoCaptureObject.release()
cv2.destroyAllWindows()


detection_model_path = 'haarcascade_frontalface_default.xml'
image_path = 'test_for_camera.jpg'

face_detection = cv2.CascadeClassifier(detection_model_path)

emotion_classifier = load_model("model.hdf5")

emotions = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']

color_frame = cv2.imread(image_path)
gray_frame = cv2.imread(image_path, 0)


cv2.imshow('Input test image', color_frame)
cv2.waitKey(10000)
cv2.destroyAllWindows()


detected_faces = face_detection.detectMultiScale(color_frame, scaleFactor=1.1, minNeighbors=5, 
                                        minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
print('Number of faces detected : ', len(detected_faces))

if len(detected_faces)>0:
    
    detected_faces = sorted(detected_faces, reverse=True, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))[0] # if more than one faces
    (fx, fy, fw, fh) = detected_faces
    
    im = gray_frame[fy:fy+fh, fx:fx+fw]
    im = cv2.resize(im, (48,48))  # the model is trained on 48*48 pixel image 
    im = im.astype("float")/255.0
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)
    
    preds = emotion_classifier.predict(im)[0]
    emotion_probability = np.max(preds)
    label = emotions[preds.argmax()]
    
    cv2.putText(color_frame, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(color_frame, (fx, fy), (fx + fw, fy + fh),(0, 0, 255), 2)

cv2.imshow('Input test image', color_frame)
cv2.imwrite('output_'+image_path.split('/')[-1], color_frame)
cv2.waitKey(10000)
cv2.destroyAllWindows()

import matplotlib.image as mpimg
img = mpimg.imread('output_for_camera.jpg')
imgplot = plt.imshow(img)
plt.show()