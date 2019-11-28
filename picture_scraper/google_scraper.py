
# imports
from selenium import webdriver
from time import sleep

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")

driver = webdriver.Chrome(chrome_options=chrome_options)
driver.get('https://google.com')

sleep(1)
