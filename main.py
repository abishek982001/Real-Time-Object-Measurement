import cv2
import objectMeasurement
import os
try:
    choice = int(input("1. Do you wish to use your webcam?\n or\n2. Do you want to use the images present in your PC?: "))
except:
    print("Invalid choice")
    exit(0)
if choice == 1:
    flag=True
elif choice == 2:
    flag =False
else:
    print("Invlid Choice")
    exit(0)
if flag:
    path=''
else:
    path = input("Enter the path of the image: ")
  
measureObject = objectMeasurement.ObjectMeasurement(flag, path)

while True:
    measureObject.run()