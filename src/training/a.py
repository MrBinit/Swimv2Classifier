import os
import time
import random
from urllib.parse import urlparse, quote_plus
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import requests
from selenium.webdriver.chrome.service import Service

# List of URLs to scrape
urls = [
    "https://www.pexels.com/search/girl/",
    "https://www.pexels.com/search/pretty/",
    "https://www.pexels.com/search/person/",
    "https://www.pexels.com/search/travel/",
    "https://www.pexels.com/search/sea/",
    "https://www.pexels.com/search/beach/",
    "https://www.pexels.com/search/beautiful/",
    "https://www.pexels.com/search/romantic/",
    "https://www.pexels.com/search/trees/",
    "https://www.pexels.com/search/outdoors/",
    "https://www.pexels.com/search/city/",
    "https://www.pexels.com/search/sky/",
    "https://www.pexels.com/search/night sky/",
    "https://www.pexels.com/search/night sky city/",
    "https://www.pexels.com/search/men/",
    "https://www.pexels.com/search/men old/",
    "https://www.pexels.com/search/men african/",
    "https://www.pexels.com/search/men white/",
    "https://www.pexels.com/search/men asian/",
    "https://www.pexels.com/search/female asian/",
    "https://www.pexels.com/search/female african/",
    "https://www.pexels.com/search/female old/",
    "https://www.pexels.com/search/nepal/",
    "https://www.pexels.com/search/india/"
]

# Base directory for saving images
base_dir = '/home/binit/classifier/data/external'

# Function to download and save an image
def download_image(img_url, save_path):
    try:
        img_data = requests.get(img_url).content
        with open(save_path, 'wb') as img_file:
            img_file.write(img_data)
    except Exception as e:
        print(f"Failed to download {img_url}. Error: {e}")

# Set up Selenium WebDriver
options = Options()
options.headless = True  # Run in headless mode
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--disable-gpu')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-blink-features=AutomationControlled')

# Specify Chrome binary location if necessary
# options.binary_location = '/usr/bin/google-chrome'

# Create a Service object
service = Service('/usr/local/bin/chromedriver')  # Update this path if necessary

driver = webdriver.Chrome(service=service, options=options)

# Loop through each URL
for url in urls:
    parsed_url = urlparse(url)
    search_term = parsed_url.path.strip('/').split('/')[-1]
    search_term_encoded = quote_plus(search_term)
    search_dir = os.path.join(base_dir, search_term.replace(' ', '_'))
    os.makedirs(search_dir, exist_ok=True)

    for page in range(1, 101):
        print(f"Scraping {search_term} - page {page}...")
        page_url = f"{parsed_url.scheme}://{parsed_url.netloc}/search/{search_term_encoded}/?page={page}"

        try:
            driver.get(page_url)
            time.sleep(random.uniform(3, 6))  # Wait for the page to load

            # Scroll to the bottom to load all images
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(random.uniform(2, 4))  # Wait for images to load

            images = driver.find_elements(By.CSS_SELECTOR, "article.photo-item img")

            if not images:
                print(f"No images found on page {page} for {search_term}.")
                break

            for idx, img in enumerate(images):
                img_url = img.get_attribute("src")
                if img_url:
                    img_name = f"{search_term}_page_{page}_image_{idx + 1}.jpg"
                    save_path = os.path.join(search_dir, img_name)
                    download_image(img_url, save_path)
                    print(f"Downloaded {img_name}")

            # Random delay to mimic human behavior
            time.sleep(random.uniform(3, 6))

        except Exception as e:
            print(f"An error occurred while processing {page_url}: {e}")
            break

driver.quit()
print("Scraping completed.")
