# code taken from https://www.linkedin.com/pulse/how-easy-scraping-data-from-linkedin-profiles-david-craven/

# imports
import numpy as np
from selenium import webdriver
from time import sleep
from parsel import Selector
import urllib

# specifies the path to the chromedriver.exe
driver = webdriver.Chrome()

# driver.get method() will navigate to a page given by the URL address
driver.get('https://www.linkedin.com')

sleep(0.5)

# locate email form by_class_name
username = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[1]/div[1]/input')

# send_keys() to simulate key strokes
username.send_keys('fibop84632@3dmail.top')

# sleep for 0.5 seconds
sleep(0.5)

# locate password form by_class_name
password = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[1]/div[2]/input')

# send_keys() to simulate key strokes
password.send_keys('TNMTMRqyY84QG-r')
sleep(0.5)

# locate submit button by_xpath
sign_in_button = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[2]/button')

# .click() to mimic button click
sign_in_button.click()

# get all the saved photo urls as a numpy array
photo_urls = np.genfromtxt('photo_urls.txt', dtype='str')
print(photo_urls)

num_images = 0
for photo_url in photo_urls:
    num_images += 1

    if num_images > 10:
        break

    # get the profile URL
    photo_url = 'https://' + photo_url
    print(photo_url)
    driver.get(photo_url)
    sel = Selector(text=driver.page_source)

    picture_url = sel.xpath('//*[@id="ember51"]')
    print(picture_url)
    if not picture_url:
        continue
    picture_url = picture_url.attrib['src']
    print(picture_url)

    # download the image
    urllib.request.urlretrieve(picture_url, "photos\\profile{}.png".format(num_images))

driver.quit()