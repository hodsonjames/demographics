from tensorflow.keras.models import load_model
import os
import numpy as np
from preprocessor import Preprocessor

races = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian'}

model = load_model('models\cnn_vgg_256.h5')
file_path = os.path.abspath(os.path.dirname(__file__))

input_folder = 'testing_data'
preprocessor = Preprocessor(file_path)
imgs_array = preprocessor.batch_preprocess(input_folder, verbose=True)

length = len(imgs_array)

x_test = np.zeros(length)
y_test = np.zeros(length)

i = 0
for img_path, img_array in imgs_array.items():
    img_name = os.path.basename(img_path)
    race = img_name.split('_')[0]
    x_test[i] = img_array
    y_test[i] = race
    i += 1

print(y_test)
print(x_test)

results = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
print('test loss, test acc:', results)
print(model.metric_names)
