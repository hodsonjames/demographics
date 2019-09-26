# 
# Class file that contains methods to detect the faces present inside 
# an image using the Histogram of Oriented Gradients (HOG) feature.
# Some of the code is sourced from the official
# dlib documentation in the public domain at:
# http://dlib.net/face_landmark_detection.py.html
# Some modifications have been made to make the code suitable for personal requirements
# 

import sys
import dlib
import time
import os
from pathlib import Path

class FaceDetector:
    
    def __init__(self, pathlist):
        """
        Instantiates a FaceDetector object 

        :param path: the path to the folder for the list images in the data set
        :type path: str
        """
        self.path = Path(pathlist).glob('*.jpg*')
        self.detector = dlib.get_frontal_face_detector()
        self.win = dlib.image_window() # to show the detected face in a GUI screen
        self.detect_faces()

    def detect_faces(self):
        """
        Iterates through all the images in the file path and detects the faces in each image,
        displaying it on the GUI with a box around the detected face

        """
        for img_path in self.path:
            path_str = str(img_path) # since path is an object
            try: 
                img, img_detec = self.detections(path_str)
                self.display_img(img, img_detec)
            except RuntimeError:
                # os.remove(path_str) # use this if you want to clean up the data and remove corrupted imgs
                continue

    def detections(self, path):
        """
        Finds the face for an indivdual image

        :return: returns two variables, img containing a numpy array of the RGB image, 
        and detections containing the specifications of the image.
        """
        img = dlib.load_rgb_image(path)
        img_detecs = self.detector(img, 1) # upsampled the image to allow more accurate detection
        return img, img_detecs

    def display_img(self, img, detections):
        """
        Displays the image on the GUI by opening a window and also overlays the image
        with the rectangular box that indicates the detected face.

        :param img: the array containing an RGB image
        :type img: numpy array
        :param detections: contains the specifications of the image
        :type detections: detections object returned by detector class
        """
        self.win.clear_overlay()
        self.win.set_image(img)
        self.win.add_overlay(detections)
        time.sleep(0.25) # use this if you want to see each img's face detection











