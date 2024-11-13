import os
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import shutil
import yaml


# Load configuration from config.yaml
with open("/home/binit/classifier/src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Retrieve paths and model details from config
destination_path = config["paths"]["raw_data"]
external_path = config["paths"]["external_dir"]

# Load the dataset
ds = load_dataset("ideepankarsharma2003/AIGeneratedImages_Midjourney")

external_real_folder = os.path.join(external_path, "real_image_hf")
external_fake_folder = os.path.join(external_path, "data_AI_hf")
raw_real_folder = os.path.join(destination_path, "real")
raw_fake_folder = os.path.join(destination_path, "AI")

# Ensure external storage folders exist for saving images
os.makedirs(external_real_folder, exist_ok=True)
os.makedirs(external_fake_folder, exist_ok=True)

# Function to convert and save images in external folder
def save_image(item, save_path):
    if isinstance(item["image"], str):  # Image is a URL
        response = requests.get(item["image"])
        image = Image.open(BytesIO(response.content)).convert("RGB")
    elif isinstance(item["image"], np.ndarray):  # Image is a NumPy array
        image = Image.fromarray(item["image"].astype(np.uint8)).convert("RGB")
    elif isinstance(item["image"], Image.Image):  # Image is already a PIL object
        image = item["image"].convert("RGB")
    else:
        raise TypeError("Unsupported image format.")

    image.save(save_path)

# Step 1: Save images in the external folders
for i, item in enumerate(ds["train"]):  # Assuming the "train" split contains all images
    label = item["label"]  # 0 for real, 1 for fake

    # Determine save path in external folder
    if label == 0:
        save_path = os.path.join(external_real_folder, f"real_{i}.jpg")
    else:
        save_path = os.path.join(external_fake_folder, f"fake_{i}.jpg")
    
    # Save the image
    save_image(item, save_path)

print("Images initially saved to 'external' folder.")

# Step 2: Move external folders to raw folders
# Ensure raw folders exist
os.makedirs(raw_real_folder, exist_ok=True)
os.makedirs(raw_fake_folder, exist_ok=True)

# Optional: Clear existing files in raw real and AI folders before moving
for filename in os.listdir(raw_real_folder):
    file_path = os.path.join(raw_real_folder, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

for filename in os.listdir(raw_fake_folder):
    file_path = os.path.join(raw_fake_folder, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Move files from external real and fake folders to raw real and fake folders
for filename in os.listdir(external_real_folder):
    shutil.move(os.path.join(external_real_folder, filename), raw_real_folder)

for filename in os.listdir(external_fake_folder):
    shutil.move(os.path.join(external_fake_folder, filename), raw_fake_folder)

print("Images moved from 'external' to 'raw' folder structure, with old files cleared.")
