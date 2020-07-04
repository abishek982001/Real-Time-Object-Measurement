import cv2
import objectMeasurement
import os

path = input("Enter path of the image: ")
measureObject = objectMeasurement.ObjectMeasurement(path)

while True:
    measureObject.run()