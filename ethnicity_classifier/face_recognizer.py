# 
# Application that recognizes the ethnicity of the faces present inside all images
# 
# Sources:
# http://dlib.net/face_landmark_detection.py.html
# https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# 
# 

import time
import os 
import glob
import cv2
from align_dlib import AlignDlib

# the dimensions in pixel that you want to crop the image to after preprocessing for  
# standardizing the images
IMG_DIM = 96 

# filepath for the folder containing all the images
# use this for the entire set of data
# input_folder_path = "C:\\Users\\yasoob\\Desktop\\Demographic Research\\data\\photos"

# use this to test only the sample
input_folder_path = "sample_photos"

# the directory you want to save the preprocessed images to
output_folder_path = "preprocessed_sample_photos"

if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

# The facial landmarks training file to identify the landmarks on the detected img
# sourced from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predic_path = 'shape_predictor_68_face_landmarks.dat'

transformer = AlignDlib(predic_path)

def main():

    img_paths = glob.glob(os.path.join(input_folder_path, '*.jpg'))

    for inp_img_path in img_paths:
        out_img_path = os.path.join(output_folder_path, os.path.basename(inp_img_path))
        output_img = preprocess(inp_img_path, out_img_path)

def preprocess(inp_path, out_path):
    """
    Detects the face present inside an image and preprocesses it by cropping, 
    aligning, and stretching the face according to the specified dimensions.
    Writes this preprocessed file onto the out_path
    :param inp_path: The path to the original image
    :type inp_path: string
    :param out_path: The output path for the preprocessed image to be stored as
    :type out_path: string
    """

    image = cv2.imread(inp_path, )

    largest_face = transformer.getLargestFaceBoundingBox(image)
    preprocessed_img = transformer.align(IMG_DIM, image, largest_face)

    cv2.imshow('image', image)
    time.sleep(1)
    cv2.imwrite(out_path, preprocessed_img)
    return preprocessed_img

main()

