import os
import glob
import cv2

folder = 'classified\\faceless'
races = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'other', 5: 'faceless',
         6: 'face_present'}

img_paths = glob.glob(os.path.join(folder, '**/*.jpg'), recursive=True)

for inp_img_path in img_paths:

    inp_img_name = os.path.basename(inp_img_path)

    # reads the image file and converts it into an RGB array
    image = cv2.imread(inp_img_path, 1)
    cv2.imshow('img', image)
    race = cv2.waitKey()
    race = race - 1 # since typing in 1, 2, 3, 4
    race = chr(race) # since waitkey() returns ASCII code
    print(race)
    os.rename(inp_img_path, folder + '\\{}\\'.format(races[int(race)]) +
              race + "_" + inp_img_name)