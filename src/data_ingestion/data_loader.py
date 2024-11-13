import os
import kagglehub
import shutil
import yaml

with open("/home/kings-college/binit/ai_classifier/src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

raw_data_path = config["paths"]["raw_data"]
dataset_id = config["dataset"]["id2"]
folder_name = config["dataset"]["folder_name2"]


# create directory for raw data if it doesnt exit

os.makedirs(raw_data_path, exist_ok= True)

# Download the dataset using kagglehub
dataset_path = kagglehub.dataset_download(dataset_id)
print(f"Downloaded to temporary path: {dataset_path}")

# Define the destination subfolder within raw
destination_folder = os.path.join(raw_data_path, folder_name)
os.makedirs(destination_folder, exist_ok=True)

# Move or copy the downloaded files to the destination subfolder
for item in os.listdir(dataset_path):
    source = os.path.join(dataset_path, item)
    destination = os.path.join(destination_folder, item)
    
    print(f"Processing item: {source} -> {destination}")
    
    # If it's a directory, move the entire directory
    if os.path.isdir(source):
        shutil.move(source, destination)
        print(f"Moved directory {source} to {destination}")
    else:
        shutil.copy2(source, destination)
        print(f"Copied file {source} to {destination}")

print(f"Dataset {dataset_id} saved to {destination_folder}")