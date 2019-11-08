# 
# Application that recognizes the ethnicity of the faces present inside all images
# 
# Sources:
# http://dlib.net/face_landmark_detection.py.html
# https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# 
#
import math
import time
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import os
import glob
import cv2
import dlib
import pickle
from align_dlib import AlignDlib # class file taken from CMUs OpenFace Project, as cited

# the dimensions in pixel that you want to crop the image to after preprocessing for
# standardizing the images
IMG_DIM = 96

# the benchmark performance time taken to preprocess a single image
BENCHMARK_TIME = 0.0249

# the factor limit by which you consider the modified code to be faster/slower than the benchmark
FASTER_FACTOR = 0.9 # 10% faster than the benchmark
SLOWER_FACTOR = 1.1 # 10% slower than the benchmark

file_path = os.path.abspath(os.path.dirname(__file__))
models_path = file_path + "/models/"

# the directory name containing all the image data
# default folder path is used to test only the sample
input_folder_name = "sample_photos"
input_folder_path = os.path.join(file_path, input_folder_name)

# the directory you want to save the preprocessed images to
# currently saving in same directory to observe the differences after preprocessing
output_folder_name = "sample_photos"
output_folder_path = os.path.join(file_path, input_folder_name)

# if the output directory does not exist, create it 
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


# The facial landmarks training file to identify the landmarks on the detected img
# sourced from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
landmark_training_name = 'shape_predictor_68_face_landmarks.dat'
predic_path = os.path.join(models_path, landmark_training_name)

# The pretrained facial recognition model, currently using dlibs ResNetv1 model
# sourced from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
facerec_model_name = 'dlib_face_recognition_resnet_model_v1.dat'
facerec_path = os.path.join(models_path, facerec_model_name)

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face, and finally the
# face recognition model.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predic_path)
facerec = dlib.face_recognition_model_v1(facerec_path)


# variables to measure performance and accuracy of code
start_time = time.time()
num_images = 0
# all corrupted images that the computer can not read and convert to RGB format
corrupted_images = 0
# images without any faces detected by the algorithm. Logos, etc. or inaccurate detection
faceless_images = 0 

# a list of the vector encodings, img names, and img attributes
feature_vecs = []
img_names = []
attributes = []
attribute_names = ['White', 'Black', 'Asian', 'Indian', 'Other']

def main():
    global num_images

    # list of all the images contained within the input_folder_path
    img_paths = glob.glob(os.path.join(input_folder_path, '**/*.jpg'), recursive=True)

    for inp_img_path in img_paths:

        if num_images >= 0:
            break

        # this gives us just the img name, such as img_name.jpg
        inp_img_name = os.path.basename(inp_img_path) 
        # this gives us just the output img name, such as img_name_preprocessed.jpg
        out_img_name = os.path.splitext(inp_img_name)[0] + "_preprocessed" + os.path.splitext(inp_img_name)[1]

        out_img_path = os.path.join(output_folder_path, out_img_name)
        pre_processed_img = preprocess(inp_img_path, out_img_path)

        num_images += 1
        print(num_images)

    # Creates and trains a KNN classifier
    # train_model()

    # test classifier on an individual image
    classify_img('68561979..jpg')



def preprocess(inp_path, out_path):
    """
    Detects the face present inside an image and preprocesses it by cropping, 
    aligning, and stretching the face according to the specified dimensions.
    Writes this preprocessed file onto the out_path
    :param inp_path: The path to the original image
    :type inp_path: string
    :param out_path: The output path for the preprocessed image to be stored as
    :type out_path: string
    :return: A list containing the aligned RGB image Shape: (imgDim, imgDim, 3) 
    and the landmark points. 
    :rtype: list[numpy.ndarray, points]		
    """
    global corrupted_images, faceless_images

    # reads the image file and converts it into an RGB array
    image = cv2.imread(inp_path, 1)

    # if image file is corrupted, removes the file from the samples
    if image is None:
        corrupted_images += 1
        os.remove(inp_path)
        return None

    # cv2.imshow('img', image)
    # cv2.waitKey(1000)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    faces = detector(image, 1)

    if len(faces) == 0:
        faceless_images += 1
        return

    # select the largest face out of all the faces in the image
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    # Get the landmarks/parts for the face in box face.
    face_landmarks = predictor(image, face)	
    aligned_img = dlib.get_face_chip(image, face_landmarks)

    # Compute the 128D vector that describes the face in aligned_img identified by
    # shape.  In general, if two face descriptor vectors have a Euclidean
    # distance between them less than 0.6 then they are from the same
    # person, otherwise they are from different people.
    face_descriptor = facerec.compute_face_descriptor(aligned_img, face_landmarks, 5)

    # add the encodings vector into the features array
    encodings_array = []
    for encoding in face_descriptor:
        encodings_array.append(encoding)
    feature_vecs.append(encodings_array)

    # add the image name to the img_names array
    img_name = os.path.basename(inp_path)
    img_names.append(img_name)

    # add the img attributes to the attributes array
    img_attributes = extract_attribute(img_name)
    attributes.append(img_attributes)

    # cv2.imshow('img', aligned_img)
    # cv2.waitKey(1000)

    cv2.imwrite(out_path, aligned_img)
    return aligned_img

def extract_attribute(img_string):
    """
    Finds the required attribute in an image name's string and then returns it
    :param img_string: the string required to extract attribute info from
    :return: an array containing the age, gender, and race of the specified image
    """
    split_string = img_string.split("_", -1)
    return split_string[2]

def train_model():
    """
    Creates and trains a  classifier that uses a set of labeled faces to
    predict the ethnicity in an unknown image
    """

    np_features = np.asarray(feature_vecs)
    np_attributes = np.asarray(attributes)
    np.savetxt('features.txt', np_features, fmt='%s')
    np.savetxt('attributes.txt', np_attributes, fmt='%s')
    train, test, train_labels, test_labels = train_test_split(feature_vecs,
                                                              attributes,
                                                              test_size=0.25,
                                                              random_state=42)

    print(np.asarray(train))
    print(np.asarray(train_labels))
    print(np.asarray(test))
    print(np.asarray(test_labels))

    # Classifier using MLP
    # classifier = MLPClassifier(solver='adam',
    #                            hidden_layer_sizes=(128, 128),
    #                            activation='relu',
    #                            max_iter=5000,
    #                            tol=1e-4)

    # Classifier using KNN
    # n_neighbors = int(round(math.sqrt(len(feature_vecs))))
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', weights='distance')

    # Classifier using Random Forest
    classifier = RandomForestClassifier(n_estimators=200)


    classifier.fit(train, train_labels)
    f = open('model.pkl', 'wb')
    pickle.dump(classifier, f)
    f.close()

    preds = classifier.predict(test)
    accuracy = metrics.accuracy_score(test_labels, preds)
    confusion_matrix = metrics.confusion_matrix(test_labels, preds)
    report = metrics.classification_report(test_labels, preds)
    print(confusion_matrix)
    print(report)
    print(accuracy)


def classify_img(img_path, clf=None):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    """

    # Load a trained KNN model (if one was passed in)
    if clf is None:
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)

    # reads the image file and finds its faces, selecting largest
    img = cv2.imread(img_path, 1)
    faces = detector(img, 1)
    face = max(faces, key=lambda rect: rect.width() * rect.height())
    face_landmarks = predictor(img, face)
    img = dlib.get_face_chip(img, face_landmarks)
    # Find encodings for faces in the test img
    encodings = facerec.compute_face_descriptor(img, face_landmarks)

    # Use the KNN model to find the best matches for the test face
    race = clf.predict([encodings])
    print(race)
    print(clf.predict_proba([encodings]))

def performanceTest():
    """
    Compares the execution time of the program to the benchmark performance time established.
    If the execution time is a lot greater than expected, prints out an error to inform the user
    that the modification of the code is inefficient and needs to be fixed.
    :param total_time: the total time consumed to run all the images in seconds
    :type total_time: floating number
    :param total_images: the total number of images processed by the program
    :type total_images: integer
    """

    total_time = time.time() - start_time
    avg_time = total_time / num_images
    dif_factor = avg_time / BENCHMARK_TIME

    print("For {} images, the total time taken was {} sec".format(num_images, total_time))
    print("The benchmark time taken for a single image is {}".format(BENCHMARK_TIME))
    print("The time taken in this execution is {}".format(avg_time))
    print("The current execution time is {} of the benchmark".format(dif_factor))


    if dif_factor >= SLOWER_FACTOR:
        print("Please check for inefficiences in the modified code and rectify them")

    if dif_factor <= FASTER_FACTOR:
        print("Good job in improving code efficiency!")

def accuracyTest():
    """
    Measures and prints out the accuracy rate of the program.
    Specifies the number of images the program found were corrupted. Also specifies
    the images in which the program detected no face, either because no face existed
    (such as logos, company profile pictures, animal pictures, etc.) or because the model 
    was inaccurate in detecting the face present.
    """
    perc_corrupted = corrupted_images / num_images
    perc_undetected = faceless_images / num_images
    total_noisy_img = corrupted_images + faceless_images
    perc_noisy_img = total_noisy_img / num_images
    perc_detected = (num_images - total_noisy_img) / num_images

    print("\n{} total images were processed".format(num_images))
    print("{} of the images were corrupted".format(perc_corrupted))
    print("{} of the images had no detected faces".format(perc_undetected))
    print("{} of the total images were noisy (corrupted or no detected faces)".format(perc_noisy_img))
    print("{} of the images had a detectable face".format(perc_detected))

main()
# performanceTest()
# accuracyTest()

