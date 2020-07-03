import cv2
import numpy as np
import utils

webcam = False
path = '1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)  # brightness
cap.set(3,1920)  # width
cap.set(4,1000)  # height
scale = 2
widthOfPaper = 210*scale
heightOfPaper = 297*scale

while True:
    if webcam:
        success, img =cap.read()
    else:
        img = cv2.imread(path)

    imgContours, finalContours = utils.getContours(img, minArea=50000, filter=4)
    if len(finalContours) != 0:
        biggest = finalContours[0][2]
        #print(biggest)
        imgWarp = utils.warpImage(img, biggest, widthOfPaper, heightOfPaper)
        imgContours2, finalContours2 = utils.getContours(imgWarp, minArea=2000, filter=4, cannyThreshold=[50,50], draw=True)
        cv2.imshow("A4", imgContours2)
        
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    cv2.waitKey(1)