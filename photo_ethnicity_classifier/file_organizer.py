# script to organize images in the folder into their racial categories

import os
import glob

folder = 'total_samples'

img_paths = glob.glob(os.path.join(folder, '**/*.jpg'), recursive=True)

for inp_img_path in img_paths:
    inp_img_name = os.path.basename(inp_img_path)
    attribute = inp_img_name.split("_", -1)[2]
    if attribute not in ['0', '1', '2', '3']:
        continue
    try:
        os.rename(inp_img_path, folder + '\\{}\\'.format(attribute) + inp_img_name)
    except:
        continue