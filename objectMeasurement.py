import cv2
import numpy as np
import utils

utils = utils.Utils()

class ObjectMeasurement:
    def __init__(self, path):
        """Parameterised constructor for initialising values"""
        self.webcam = False
        self.path = path
        self.cap = cv2.VideoCapture(0)
        self.cap.set(10,160)  # brightness of the camera
        self.cap.set(3,1920)  # width of the camera frame
        self.cap.set(4,1000)  # height of the camera frame
        self.scale = 2
        self.widthOfPaper = 210*self.scale
        self.heightOfPaper = 297*self.scale
        
    def run(self):
        """Function to start the webcam or load the image according to users choice and perform the 
        required calculations for a single frame"""
        if self.webcam:
            success, img =self.cap.read()
        else:
            img = cv2.imread(self.path) 
        imgContours, finalContours = utils.getContours(img, minArea=50000, filter=4)
        if len(finalContours) != 0:
            biggest = finalContours[0][2]
            #print(biggest)
            imgWarp = utils.warpImage(img, biggest, self.widthOfPaper, self.heightOfPaper)
            imgContours2, finalContours2 = utils.getContours(imgWarp, minArea=2000, filter=4, cannyThreshold=[50,50])
            for object in finalContours2:
                cv2.polylines(imgContours2, [object[2]], True, (0,255,0), 2)
                newPoints = utils.reorder(object[2])
                newWidth = round((utils.findDistance(newPoints[0][0]//self.scale, newPoints[1][0]//self.scale)/10),1)  # /10 to convert it to 
                newHeight = round((utils.findDistance(newPoints[0][0]//self.scale, newPoints[2][0]//self.scale))/10,1)
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