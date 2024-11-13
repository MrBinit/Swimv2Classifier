import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

# Load configuration
with open("/home/binit/classifier/src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Retrieve paths
processed_data_path = config["paths"]["processed_data"]
split_data_path = config["paths"]["split_data"]

# Convert split ratios from strings to floats
train_ratio = float(config["splits"]["train"])
test_ratio = float(config["splits"]["test"])
validation_ratio = float(config["splits"]["validation"])

# Ensure the sum of splits equals 1.0
if not abs(train_ratio + test_ratio + validation_ratio - 1.0) < 1e-6:
    raise ValueError("The sum of train, test, and validation ratios must equal 1.0.")

# Ensure directories exist for train, test, and validation
def make_dir_structure(base_path):
    for split in ["train", "validation", "test"]:
        for category in ["real", "AI"]:
            os.makedirs(os.path.join(base_path, split, category), exist_ok=True)

# Function to clear existing files in split_data_path
def clear_split_data(base_path):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        print(f"Cleared existing split data at {base_path}")
        
# clearing the old data 
clear_split_data(split_data_path)
# Create directory structure
make_dir_structure(split_data_path)

# Collect all real and AI image files
real_files = [os.path.join(processed_data_path, "real", f)
              for f in os.listdir(os.path.join(processed_data_path, "real"))
              if f.endswith(".pt")]
ai_files = [os.path.join(processed_data_path, "AI", f)
            for f in os.listdir(os.path.join(processed_data_path, "AI"))
            if f.endswith(".pt")]

def split_and_save_data(files, base_path, category):
    # Split data into train and temp (test + validation)
    train_files, temp_files = train_test_split(
        files, test_size=(validation_ratio + test_ratio), random_state=42
    )
    # Split temp into validation and test
    val_files, test_files = train_test_split(
        temp_files, test_size=test_ratio / (validation_ratio + test_ratio), random_state=42
    )

    # Function to copy files to the designated folder
    def copy_files(file_list, split_name):
        for f in file_list:
            shutil.copy(f, os.path.join(base_path, split_name, category))

    # Copy files to respective directories
    copy_files(train_files, "train")
    copy_files(val_files, "validation")
    copy_files(test_files, "test")

    print(f"{category.capitalize()} files: {len(train_files)} for training, "
          f"{len(val_files)} for validation, {len(test_files)} for testing.")

# Split and save data for both categories
split_and_save_data(real_files, split_data_path, "real")
split_and_save_data(ai_files, split_data_path, "AI")

print("Data has been split and saved successfully.")
