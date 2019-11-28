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
username.send_keys('gowewa9161@hide-mail.net')
# username.send_keys('bohob55303@net3mail.com')
# username.send_keys('relon67160@mailhub24.com')

# sleep for 0.5 seconds
sleep(0.5)

# locate password form by_class_name
password = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[1]/div[2]/input')

# send_keys() to simulate key strokes
password.send_keys('Rcxph8w/$#R8Mus')
# password.send_keys('pL$7ZVEgyqXw*Pw')
# password.send_keys('w-US6ir82e.TJ4R')
sleep(0.5)

# locate submit button by_xpath
sign_in_button = driver.find_element_by_xpath('/html/body/nav/section[2]/form/div[2]/button')

# .click() to mimic button click
sign_in_button.click()

# get all the saved photo urls as a numpy array
photo_urls = np.genfromtxt('photo_urls.txt', dtype='str', max_rows=1000)
photo_urls = photo_urls[149:]
print(photo_urls)

num_images = 0
wrong_tag = 0
unavailable_profile = 0
no_picture = 0
actual_images = 0
for photo_url in photo_urls:
    sleep(1)
    if num_images >= 150:
        break

    num_images += 1
    print(num_images)

    # get the profile URL
    photo_url = 'https://' + photo_url
    print(photo_url)
    driver.get(photo_url)
    sel = Selector(text=driver.page_source)

    picture_url = sel.xpath('//*[@id="ember51"]')
    if not picture_url:
        picture_url = sel.xpath('/html/body/main/section[1]/section/section[1]/div/img')
    print(picture_url)
    if not picture_url:
        wrong_tag += 1
        continue

    try:
        picture_url = picture_url.attrib['src']
    except:
        unavailable_profile += 1
        continue

    print(picture_url)

    if picture_url == 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'\
            or picture_url == 'https://static-exp1.licdn.com/sc/h/djzv59yelk5urv2ujlazfyvrk':
        no_picture += 1
        continue

    # download the image
    urllib.request.urlretrieve(picture_url, "photos\\profile{}.png".format(150 + num_images))
    actual_images += 1

driver.quit()



incorrect_perc = round(wrong_tag/num_images, 3)
unavailable_perc = round(unavailable_profile/num_images, 3)
no_pic_perc = round(no_picture/num_images, 3)
obtained_img_perc = round(actual_images/num_images, 3)

print('Total images: {}, Incorrectly Tagged: {}, Unavailable Profiles: {}, '
      'No Picture Uploaded: {}, Obtained Images: {}, '
      '\nIncorrect Percentage: {}, Unavailable Percentage: {}, No Picture Percentage: {}, '
      'Obtained Images Percentage: {}'.format(num_images, wrong_tag, unavailable_profile,
                                              no_picture, actual_images, incorrect_perc,
                                              unavailable_perc, no_pic_perc, obtained_img_perc))