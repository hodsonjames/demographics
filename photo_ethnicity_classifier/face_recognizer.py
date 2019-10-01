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
from align_dlib import AlignDlib # class file taken from CMUs OpenFace Project, as cited

# the dimensions in pixel that you want to crop the image to after preprocessing for  
# standardizing the images
IMG_DIM = 96 

# the benchmark performance time taken to preprocess a single image
BENCHMARK_TIME = 0.04731

# the factor limit by which you consider the modified code to be faster/slower than the benchmark
FASTER_FACTOR = 0.9 # 10% faster than the benchmark
SLOWER_FACTOR = 1.1 # 10% slower than the benchmark

# the directory name containing all the image data
# default folder path is used to test only the sample
input_folder_name = "sample_photos"
input_folder_path = os.path.join(os.path.dirname(__file__), input_folder_name)

# the directory you want to save the preprocessed images to
# currently saving in same directory to observe the differences after preprocessing
output_folder_name = "sample_photos"
output_folder_path = os.path.join(os.path.dirname(__file__), input_folder_name)

# if the output directory does not exist, create it 
if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

# The facial landmarks training file to identify the landmarks on the detected img
# sourced from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
landmark_training_name = 'shape_predictor_68_face_landmarks.dat'
predic_path = os.path.join(os.path.dirname(__file__), landmark_training_name)

transformer = AlignDlib(predic_path)

def main():

    # variables to measure performance of code
    start_time = time.time()
    num_images = 0

    # list of all the images contained within the input_folder_path
    img_paths = glob.glob(os.path.join(input_folder_path, '*.jpg'))

    for inp_img_path in img_paths:
        num_images += 1

        # this gives us just the img name, such as img_name.jpg
        inp_img_name = os.path.basename(inp_img_path) 
        # this gives us just the output img name, such as img_name_preprocessed.jpg
        out_img_name = os.path.splitext(inp_img_name)[0] + "_preprocessed" + os.path.splitext(inp_img_name)[1]

        out_img_path = os.path.join(output_folder_path, out_img_name)
        output_img = preprocess(inp_img_path, out_img_path)

    time_taken = time.time() - start_time
    performanceTest(time_taken, num_images)

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

    # reads the image file and converts it into an RGB array
    image = cv2.imread(inp_path, )
    
    # if image file is corrupted, removes the file from the samples
    if image is None:
        os.remove(inp_path)
        return

    largest_face = transformer.getLargestFaceBoundingBox(image)
    preprocessed_img = transformer.align(IMG_DIM, image, largest_face)

    cv2.imwrite(out_path, preprocessed_img)
    return preprocessed_img

def performanceTest(total_time, total_images):
    """
    Compares the execution time of the program to the benchmark performance time established.
    If the execution time is a lot greater than expected, prints out an error to inform the user
    that the modification of the code is inefficient and needs to be fixed.
    :param total_time: the total time consumed to run all the images in seconds
    :type total_time: floating number
    :param total_images: the total number of images processed by the program
    :type total_images: integer
    """

    avg_time = total_time / total_images
    dif_factor = avg_time / BENCHMARK_TIME

    print("The benchmark time taken for a single image is {}".format(BENCHMARK_TIME))
    print("The time taken in this execution is {}".format(avg_time))
    print("The current execution time is {} that of the benchmark".format(dif_factor))


    if dif_factor >= SLOWER_FACTOR:
        print("Please check for inefficiences in the modified code and rectify them")

    if dif_factor <= FASTER_FACTOR:
        print("Good job in improving code efficiency!")

main()

