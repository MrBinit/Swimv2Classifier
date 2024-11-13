import os
import shutil

# Define source directories and the destination directory
source_paths = [
    "/home/kings-college/binit/ai_classifier/data/raw/dataset1/RealArt/RealArt",
    "/home/kings-college/binit/ai_classifier/data/raw/dataset2/Midjourney_Exp2/test/REAL",
    "/home/kings-college/binit/ai_classifier/data/raw/dataset2/Midjourney_Exp2/train/REAL",
    "/home/kings-college/binit/ai_classifier/data/raw/dataset2/Midjourney_Exp2/valid/REAL",
    "/home/kings-college/binit/ai_classifier/data/raw/dataset3/test/real",
    "/home/kings-college/binit/ai_classifier/data/raw/dataset3/train/real"
]

destination_path = "/home/kings-college/binit/ai_classifier/data/raw/real"

# Ensure the destination directory exists
os.makedirs(destination_path, exist_ok=True)

# Move all files from each source directory to the destination directory
for source_path in source_paths:
    if not os.path.exists(source_path):
        print(f"Source path does not exist: {source_path}")
        continue

    for filename in os.listdir(source_path):
        source_file = os.path.join(source_path, filename)
        destination_file = os.path.join(destination_path, filename)

        # Move the file
        shutil.move(source_file, destination_file)
        print(f"Moved: {source_file} -> {destination_file}")

print("All files moved successfully.")