# 
# Application that recognizes the ethnicity of the faces present inside an image 
# 
# Sources:
# http://dlib.net/face_landmark_detection.py.html
# https://hackernoon.com/building-a-facial-recognition-pipeline-with-deep-learning-in-tensorflow-66e7645015b8
# https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78
# 
# Some modifications have been made to make the code suitable for personal requirements
# 

import face_detector

# list of image filepaths
# use this for the entire set of data
#pathlist = "C:\\Users\\yasoob\\Desktop\\Demographic Research\\data\\photos"

# use this to test only the sample
pathlist = "sample_photos"

face_detector = face_detector.FaceDetector(pathlist) # runs the detector on all the images in pathlist

