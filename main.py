import cv2
import objectMeasurement

measureObject = objectMeasurement.ObjectMeasurement()

while True:
    img = measureObject.run()
    img = cv2.resize(img, (0,0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    cv2.waitKey(1)
