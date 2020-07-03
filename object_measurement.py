import cv2
import numpy as np
import utils

webcam = False
path = '1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)  # brightness
cap.set(3,1920)  # width
cap.set(4,1000)  # height

while True:
    if webcam:
        success, img =cap.read()
    else:
        img = cv2.imread(path)

    utils.getContours(img, showCanny=True)
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    cv2.waitKey(1)