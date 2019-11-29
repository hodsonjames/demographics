import dlib
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
from preprocessor import Preprocessor

races = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian'}

model = load_model('models\cnn_vgg_256.h5')
file_path = os.path.abspath(os.path.dirname(__file__))
print(1)

input_folder = 'original'
output_folder = 'classified'
preprocessor = Preprocessor(file_path)
imgs_array = preprocessor.batch_preprocess(input_folder)
print(imgs_array)
print(2)

for img_path, img_array in imgs_array.items():
    img_name = os.path.basename(img_path)
    print(img_name)
    if img_array == 'faceless' or img_array == 'corrupted':
        os.rename(img_path, output_folder + '\\{}\\'.format(img_array) + img_name)
        continue

    prediction = model.predict(img_array, verbose=1)
    predicted_class = prediction.argmax(axis=-1)
    prediction = np.around(prediction[0], decimals=3)
    os.rename(img_path, output_folder + '\\{}\\'.format(races[predicted_class]) + img_name)

# print('White: ' + str(prediction[0]) +
#       '\nBlack: ' + str(prediction[1]) +
#       '\nAsian: ' + str(prediction[2]) +
#       '\nIndian: ' + str(prediction[3]))

