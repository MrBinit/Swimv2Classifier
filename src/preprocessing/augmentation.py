import os
import numpy as np
from PIL import Image
import albumentations as A
import yaml

with open("/home/binit/classifier/src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)
augmented_data_path = config["paths"]["augmented_data"]

# Define the augmentation pipeline
albumentations_augmentations = A.Compose([
    A.AdvancedBlur(blur_limit=(3, 15), sigma_x_limit=(0.2, 1.0), sigma_y_limit=(0.2, 1.0), p=0.1),
    A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.1),
    A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.05),
    A.ChannelShuffle(p=0.05),
    A.Downscale(scale_min=0.75, scale_max=0.95, p=0.1),
    A.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.5), p=0.05),
    A.Rotate(limit=5, p=0.1),
    A.Transpose(p=0.05),
    A.Flip(p=0.2),
    A.RandomToneCurve(scale=0.2, p=0.1),
    A.Equalize(mode='cv', by_channel=True, p=0.1),
    A.GaussNoise(var_limit=(5.0, 20.0), p=0.1)
])

def apply_augmentation(image):
    augmented = albumentations_augmentations(image=image)
    return augmented["image"]

def process_and_augment_images(source_dirs, target_root_dir):
    for source_dir in source_dirs:
        subdir_name = os.path.basename(source_dir.rstrip('/'))
        target_dir = os.path.join(target_root_dir, subdir_name)

        # Create the target subdirectory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir(source_dir):
            source_filepath = os.path.join(source_dir, filename)
            target_filepath = os.path.join(target_dir, filename)

            if os.path.exists(target_filepath):
                print(f"Skipping already augmented images : {target_filepath}")
                continue
            
            # Check if the file is an image
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    with Image.open(source_filepath) as img:
                        # Convert image to RGB if it's not
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_array = np.array(img)
                        augmented_img_array = apply_augmentation(img_array)
                        augmented_img = Image.fromarray(augmented_img_array)
                        target_filepath = os.path.join(target_dir, filename)
                        augmented_img.save(target_filepath)

                        print(f"Processed and saved: {target_filepath}")

                except Exception as e:
                    print(f"Failed to process {source_filepath}: {e}")

if __name__ == "__main__":
    source_directories = [
        "/home/binit/classifier/data/raw/AI",
        "/home/binit/classifier/data/raw/real"
    ]
    target_root_directory = augmented_data_path

    # Process and augment images
    process_and_augment_images(source_directories, target_root_directory)
