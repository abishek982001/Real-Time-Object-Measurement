import cv2
import numpy as np

class Utils:
    def getContours(self, img, cannyThreshold=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
        try:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except:
            print("Invalid image path")
            exit(0)
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
                        finalContours.append([len(approx), area, approx, bbox, contour])
                else:
                    finalContours.append([len(approx), area, approx, bbox, contour])
            finalContours = sorted(finalContours, key=lambda x:x[1], reverse=True)
            if draw:
                for contour in finalContours:
                    cv2.drawContours(img, contour[4], -1, (0,0,255), 3)
        return img, finalContours

    def reorder(self, points):
        shape = points.shape
        newPoints = np.zeros_like(points)
        points = points.reshape((shape[0],shape[2]))
        add = points.sum(1)
        newPoints[0] = points[np.argmin(add)]
        newPoints[3] = points[np.argmax(add)]
        diff = np.diff(points, axis=1)
        newPoints[1] = points[np.argmin(diff)]
        newPoints[2] = points[np.argmax(diff)]
        return newPoints

    def warpImage(self, img, points, width, height, pad=20):
        # print(points)
        # print(reorder(points))
        points = self.reorder(points)
        pst1 = np.float32(points)
        pst2 = np.float32([[0,0], [width,0], [0,height], [width,height]])
        matrix = cv2.getPerspectiveTransform(pst1, pst2)
        imgWarp = cv2.warpPerspective(img, matrix, (width,height))
        imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad] # To remove the edges in the A4 paper
        return imgWarp

    def findDistance(self, pts1, pts2):
        return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5
