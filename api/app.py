import os
import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from transformers import AutoImageProcessor, Swinv2Model
import torch
import numpy as np

# Load configuration from config.yaml
config_path = "/home/binit/classifier/src/config/config.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Model configuration
pretrained_model = config.get("models", {}).get("model_name", "microsoft/swinv2-tiny-patch4-window8-256")
image_extensions = tuple(config.get("models", {}).get("image_extensions", [".jpg", ".jpeg", ".png"]))

# Initialize the image processor and model
try:
    image_processor = AutoImageProcessor.from_pretrained(pretrained_model)
    model = Swinv2Model.from_pretrained(pretrained_model)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI
app = FastAPI()

# Preprocessing function
def preprocess_image(image: Image.Image) -> torch.Tensor:
    image_np = np.array(image.convert("RGB"))
    inputs = image_processor(image_np, return_tensors="pt")
    with torch.no_grad():
        features = model(**inputs).last_hidden_state
    return features

# Endpoint for preprocessing the image
@app.post("/preprocess/")
async def preprocess(image_file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    if not image_file.filename.lower().endswith(image_extensions):
        raise HTTPException(status_code=400, detail="File format not supported. Please upload an image file.")

    try:
        # Open and preprocess the image
        image = Image.open(image_file.file)
        features = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    # Convert features to a list for JSON serialization
    features_list = features.squeeze().tolist()

    return {"features": features_list}
