import cv2
import numpy as np

def getContours(img, cannyThreshold=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cannyThreshold[0], cannyThreshold[1])
    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThreshold = cv2.erode(imgDil, kernel, iterations=2)
    if showCanny:
        cv2.imshow("Canny", imgThreshold)
    contours, heirachy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > minArea:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append(len(approx), area, approx, bbox, contour)
            else:
                finalContours.append((len(approx), area, approx, bbox, contour))
        finalContours = sorted(finalContours, key=lambda x:x[1], reverse=True)
        if draw:
            for contour in finalContours:
                cv2.drawContours(img, contour[4], -1, (0,0,255), 3)
    return img, finalContours