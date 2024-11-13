import os
from PIL import Image
import numpy as np


raw_data_path = '/home/binit/classifier/data/split_data/train'
preprocessed_data_path = '/home/binit/classifier/data/feature_extraction'


# Create directories if they don't exist
os.makedirs(os.path.join(preprocessed_data_path, 'AI'), exist_ok=True)
os.makedirs(os.path.join(preprocessed_data_path, 'real'), exist_ok=True)

# Minimal Preprocessing: Resize and normalize images
def preprocess_image(image_path, output_path, target_size=(224, 224)):
    try:
        image = Image.open(image_path).convert("RGB")
        # Resize the image with LANCZOS filter for high-quality downsampling
        image = image.resize(target_size, Image.LANCZOS)
        # Convert to numpy array and normalize to [0, 1] range
        image_array = np.array(image) / 255.0
        # Convert back to image and save
        preprocessed_image = Image.fromarray((image_array * 255).astype(np.uint8))
        preprocessed_image.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process images in both 'AI' and 'real' folders
for category in ['AI', 'real']:
    input_folder = os.path.join(raw_data_path, category)
    output_folder = os.path.join(preprocessed_data_path, category)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpeg', '.jpg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            preprocess_image(input_path, output_path)

print("Minimal preprocessing completed and saved to preprocessed folder.")