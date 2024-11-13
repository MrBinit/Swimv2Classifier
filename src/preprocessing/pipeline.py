import os
import subprocess
import yaml

with open("/home/binit/classifier/src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)


raw_data_path = config["paths"]["raw_data"]
processed_data_path = config["paths"]["processed_data"]
split_data_path = config["paths"]["split_data"]

# def hugging_face_data_loader():
#     print(f"Dowloading huggingface data")
#     subprocess.run(["python", "huggingface_dataloader.py"])
#     print("Downloaded huggingface data")

def extension_converter():
    print("Running the extension converter")
    subprocess.run(["python", "extension_converter.py"])
    print("Extension converter done.")

def augmentation():
    print("Running augmentation process")
    subprocess.run(["python", "augmentation.py"])
    print("augmentation done")

def run_preprocessing():
    print("Running preprocessing...")
    subprocess.run(["python", "preprocess.py"])
    print("Preprocessing completed.")

def run_splitting():
    print("Running data splitting...")
    subprocess.run(["python", "split_data.py"])
    print("Data splitting completed.")

def run_pipeline():
    # hugging_face_data_loader()
    extension_converter()
    augmentation()
    run_preprocessing()
    run_splitting()
    print(f"Pipleline execution completed. ")

if __name__ == "__main__":
    run_pipeline()

