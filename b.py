from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image


load_directory = "/home/binit/classifier/hf_model"
processor = AutoImageProcessor.from_pretrained(load_directory)
model = AutoModelForImageClassification.from_pretrained(load_directory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open("/home/binit/classifier/data/raw/real/9U9A9073.JPG")

inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
id2label = model.config.id2label
predicted_label = id2label[predicted_class_idx]

print(f"The image is classified as: {predicted_label}")
