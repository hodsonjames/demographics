# 
# Application that detects the faces present inside an image using the Histogram of 
# Oriented Gradients (HOG) feature.
# Most of the code is sourced from the official
# dlib documentation in the public domain at:
# http://dlib.net/face_landmark_detection.py.html
# Some modifications have been made to make the code suitable for personal requirements
# 

import sys
import dlib
import time
import os
from pathlib import Path

detector = dlib.get_frontal_face_detector()
win = dlib.image_window() #to show the detected face in a GUI screen

#list of image filepaths
pathlist = Path("C:\\Users\\yasoob\\Desktop\\Demographic Research\\data\\photos").glob('*.jpg*')

for path in pathlist:
    path_str = str(path) #since path is an object
    print(path_str)
    try: 
        img = dlib.load_rgb_image(path_str)
        detections = detector(img, 1) #upsampled the image to allow greater detection
        print("Num of faces : {}".format(len(detections)))
        for i, d in enumerate(detections):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, d.left(), d.top(), d.right(), d.bottom()))
        win.clear_overlay()
        win.set_image(img)
        win.add_overlay(detections)
        #time.sleep(1) # use this if you want to see each img's face detection
    except RuntimeError:
        #os.remove(path_str) # use this if you want to clean up the data and remove corrupted imgs
        continue
    





