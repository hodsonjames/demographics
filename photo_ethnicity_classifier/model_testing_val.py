from tensorflow.keras.models import load_model
import os
import numpy as np
from preprocessor import Preprocessor
import csv

races = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian'}

model = load_model('models/cnn_vgg_256.h5')
file_path = os.path.abspath(os.path.dirname(__file__))

f = open('fairface_label_val.csv')
csv_f = csv.reader(f)

race_mapping = {}
race_groups = {'White': 0, 'Black': 1, 'East Asian': 2,
                'Southeast Asian': 2, 'Indian': 3,
                'Middle Eastern': 3, 'Latino_Hispanic': 4}
first_row = False

for row in csv_f:
    if not first_row:
        first_row = True
        continue
    age = row[1]
    if age == '3-9' or age == '60-69':
        continue
    name = os.path.basename(row[0])
    race_mapping[name] = race_groups[row[3]]

print(race_mapping)

input_folder = 'val'
preprocessor = Preprocessor(file_path)
imgs_array = preprocessor.batch_preprocess(input_folder, verbose=True)

correct = 0
incorrect = 0
total = 0
result = np.zeros((5,5))
totals = np.zeros(5)

for img_path, img_array in imgs_array.items():
    if img_array == 'faceless' or img_array == 'corrupted':
        continue
    img_name = os.path.basename(img_path)
    if img_name not in race_mapping:
        continue
    total += 1
    race = race_mapping[img_name]
    print(img_name, race)
    race = int(race)
    prediction = model.predict(img_array)
    predicted_class = prediction.argmax(axis=-1)
    prediction = np.around(prediction[0], decimals=3)
    if race == predicted_class:
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






