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
from pathlib import Path

detector = dlib.get_frontal_face_detector()
win = dlib.image_window() #to show the detected face in a GUI screen

#list of image filepaths
pathlist = Path('sample_photos').glob('*.jpg*')

for path in pathlist:
    path_str = str(path) #since path is an object
    print(path_str)
    img = dlib.load_rgb_image(path_str)
    detections = detector(img, 1) #upsampled the image to allow greater detection
    print("Num of faces : {}".format(len(detections)))
    for i, d in enumerate(detections):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(detections)
    dlib.hit_enter_to_continue()





