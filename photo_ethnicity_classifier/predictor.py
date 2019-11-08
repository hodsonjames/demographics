import dlib
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np

races = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian'}

model = load_model('models\cnn_vgg_256.h5')
file_path = os.path.abspath(os.path.dirname(__file__))

aligned_img = aligned_img.reshape((-1, 150, 150, 3))
aligned_img = aligned_img.astype(np.float32)

prediction = model.predict(aligned_img, verbose=1)

prediction = np.around(prediction[0], decimals=3)

predicted_class = prediction.argmax(axis=-1)
print(races[predicted_class])

print('White: ' + str(prediction[0]) +
      '\nBlack: ' + str(prediction[1]) +
      '\nAsian: ' + str(prediction[2]) +
      '\nIndian: ' + str(prediction[3]))

