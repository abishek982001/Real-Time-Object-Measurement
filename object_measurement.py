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
        imgContours2, finalContours2 = utils.getContours(imgWarp, minArea=2000, filter=4, cannyThreshold=[50,50])
        for object in finalContours2:
            cv2.polylines(imgContours2, [object[2]], True, (0,255,0), 2)
            newPoints = utils.reorder(object[2])
            newWidth = round((utils.findDistance(newPoints[0][0]//scale, newPoints[1][0]//scale)/10),1)  # /10 to convert it to 
            newHeight = round((utils.findDistance(newPoints[0][0]//scale, newPoints[2][0]//scale))/10,1)
            cv2.arrowedLine(imgContours2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[1][0][0], newPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours2, (newPoints[0][0][0], newPoints[0][0][1]), (newPoints[2][0][0], newPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = object[3]
            cv2.putText(imgContours2, '{}cm'.format(newWidth), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
            cv2.putText(imgContours2, '{}cm'.format(newHeight), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)       
        cv2.imshow("A4", imgContours2)

    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    cv2.waitKey(1)

