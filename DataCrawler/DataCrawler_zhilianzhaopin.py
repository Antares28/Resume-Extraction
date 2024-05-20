from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import time


# Constants
base_url = 'https://www.gerenjianli.com'
headers = {
    'Connection': 'close',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
}


# Initial setup for WebDriver
driver = webdriver.Chrome()


# Function to ensure URL format
def format_url(href):
    return href if href.startswith('http') else f"{base_url}{href}"


# Collect detail URLs
detail_urls = set()
for page in range(7, 10):
    driver.get(f"{base_url}/moban/index_{page}.html")
    links = driver.find_elements(By.CSS_SELECTOR, '.list_boby .prlist li div a')
    detail_urls.update({format_url(link.get_attribute('href')) for link in links})


# Download files from detail URLs
for link in detail_urls:
    driver.get(link) 
    download_button = driver.find_element(By.CSS_SELECTOR, ".donwurl2 a")
    download_button.click()  
    time.sleep(10)

driver.quit()