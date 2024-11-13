import os
import yaml
from transformers import AutoImageProcessor, Swinv2Model
from PIL import Image
import torch
import numpy as np

# Load configuration from config.yaml
with open("/home/binit/classifier/src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Retrieve paths and model details from config
raw_data_path = config["paths"]["raw_data"]
augmented_data_path = config["paths"]["augmented_data"]
processed_data_path = config["paths"]["processed_data"]
pretrained_model = config["models"]["model_name"]
image_extensions = tuple(config["models"]["image_extensions"])
print("Loaded paths and model successfully")

# Initialize the image processor and model
image_processor = AutoImageProcessor.from_pretrained(pretrained_model)
model = Swinv2Model.from_pretrained(pretrained_model)

# Function to ensure processed folders exist
def ensure_processed_folders():
    for folder_name in ["real", "AI"]:
        os.makedirs(os.path.join(processed_data_path, folder_name), exist_ok=True)
    print("Ensured 'real' and 'AI' folders exist in the processed data path")

# Function to process and save images
def process_and_save_images(source_path, folder_name, is_augmented= False):
    print(f"Processing {'augmented' if is_augmented else 'raw'} images in '{folder_name}' from {source_path}")

    processed_folder_path = os.path.join(processed_data_path, folder_name)

    # Check if the source folder exists
    if not os.path.exists(source_path):
        print(f"Source folder '{source_path}' doesn't exist. Skipping.")
        return
    
    # Process each file in the source folder
    for filename in os.listdir(source_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(source_path, filename)

            if is_augmented:
                base, ext = os.path.splitext(filename)
                filename= f"{base}_aug{ext}"

            save_path = os.path.join(processed_folder_path, f"{filename}.pt")

            # Skip if already processed
            if os.path.exists(save_path):
                print(f"Skipping already processed image: {filename}")
                continue

            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image)

                # Process image
                inputs = image_processor(image_np, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs).last_hidden_state

                # Save the processed tensor
                torch.save(outputs, save_path)
                print(f"Processed and saved: {save_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        else:
            print(f"Skipped non-image file: {filename}")

# Ensure folders exist in the processed data path
ensure_processed_folders()

# Process the "real" and "AI" folders in both raw_data_path and augmented_data_path
for folder_name in ["real", "AI"]:
    raw_folder_path = os.path.join(raw_data_path, folder_name)
    augmented_folder_path = os.path.join(augmented_data_path, folder_name)

    # # Process raw images
    process_and_save_images(raw_folder_path, folder_name, is_augmented=False)

    # Process augmented images
    process_and_save_images(augmented_folder_path, folder_name, is_augmented=True)

print("All processing completed.")
