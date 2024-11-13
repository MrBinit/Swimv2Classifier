import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from PIL import Image
from transformers import AutoImageProcessor
import numpy as np

# Load and modify the model
num_classes = 2  # Adjust based on your specific use case (e.g., AI vs. real)
model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)

# Update the head of the model to match the number of classes
model.head = nn.Sequential(
    nn.Dropout(0.5),  # Adjust dropout rate if needed
    nn.Linear(model.head.in_features, num_classes)
)

# Load the saved model weights
model_path = '/home/binit/classifier/models/final_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

# Set the model to evaluation mode
model.eval()

# Initialize the image processor
image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

# Define the function for preprocessing and classifying an image
def preprocess_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    inputs = image_processor(image_np, return_tensors="pt")
    return inputs['pixel_values']  # Returns tensor with batch dimension

def classify_image(image_path: str) -> str:
    # Preprocess the image
    image_tensor = preprocess_image(image_path)  # Keep batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = nn.functional.softmax(output, dim=1)  # Apply softmax to get probabilities
        confidence, prediction = torch.max(probabilities, 1)  # Get the class index and confidence

    # Map the prediction to the class name
    class_map = {0: "AI", 1: "real"}
    predicted_class = class_map[prediction.item()]
    confidence_score = confidence.item()

    # Get probabilities for both classes
    ai_probability = probabilities[0, 0].item()
    real_probability = probabilities[0, 1].item()

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "ai_probability": ai_probability,
        "real_probability": real_probability
    }

# Example usage
image_path = "/home/binit/classifier/data/raw/real/top-100-photos-of-the-year-2015-094.jpeg"  # Replace with your image path
result = classify_image(image_path)
print(f"The image is classified as: {result['predicted_class']}")
print(f"Confidence score: {result['confidence_score']:.4f}")
print(f"AI probability: {result['ai_probability']:.4f}")
print(f"Real probability: {result['real_probability']:.4f}")
