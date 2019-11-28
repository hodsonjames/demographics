
# imports
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from parsel import Selector
from urllib.request import *
import json
import os

download_path = "photos"

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")

driver = webdriver.Chrome(chrome_options=chrome_options)

# get all the saved photo urls as a numpy array
photo_urls = np.genfromtxt('photo_urls.txt', dtype='str', max_rows=1000)

i = 0
no_images = 0
for photo_url in photo_urls:
    print(i)
    if i > 60:
        break
    sleep(0.5)
    photo_id = photo_url.split('/')[-1]

    driver.get('https://www.google.com/search?q='+photo_id+'&source=lnms&tbm=isch')

    # total images in the first page
    imgs = 	driver.find_elements_by_xpath('//div[contains(@class,"rg_meta")]')

    # create a folder to save the images
    if len(imgs) > 0:
        img_folder = download_path + '\\{}\\'.format(photo_id)
        if not os.path.exists(img_folder):
            os.mkdir(img_folder)
        else:
            continue

    img_count = 0
    for img in imgs:
        img_url = json.loads(img.get_attribute('innerHTML'))["ou"]
        print(img_url)
        try:
            urlretrieve(img_url, img_folder + '\\{}.png'.format(img_count))
        except:
            continue
        img_count += 1
        if img_count >= 5:
            break

    i += 1

driver.quit()


