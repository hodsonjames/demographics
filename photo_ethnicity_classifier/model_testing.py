from tensorflow.keras.models import load_model
import os
import numpy as np
from preprocessor import Preprocessor

races = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian'}

model = load_model('models\cnn_aug_128.h5')
file_path = os.path.abspath(os.path.dirname(__file__))

input_folder = 'testing_data'
preprocessor = Preprocessor(file_path)
imgs_array = preprocessor.batch_preprocess(input_folder)

correct = 0
incorrect = 0
total = 0
result = np.zeros((5,5))
totals = np.zeros(5)

for img_path, img_array in imgs_array.items():
    total += 1
    img_name = os.path.basename(img_path)
    race = img_name.split('_')[0]
    race = int(race)
    prediction = model.predict(img_array)
    predicted_class = prediction.argmax(axis=-1)
    prediction = np.around(prediction[0], decimals=3)
    if race == predicted_class[0]:
        correct += 1
    else:
        incorrect += 1
    result[race][predicted_class] += 1
    totals[race] += 1

print('Correct: {} Incorrect: {}'.format(correct/total, incorrect/total))
print(result)
for i in range(5):
    for j in range(5):
        result[i][j] = np.around(result[i][j]/totals[i], decimals=3)

print(result)
print(totals)






