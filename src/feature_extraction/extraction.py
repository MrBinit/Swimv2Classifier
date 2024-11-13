import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
from scipy.fft import fft2
from PIL import Image, UnidentifiedImageError

# Function to load image and convert to grayscale
def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return np.array(image)
    except UnidentifiedImageError:
        print(f"Unidentified image found and deleted: {image_path}")
        os.remove(image_path) 
        return None 


def extract_texture_features(image):
    gray_image = rgb2gray(image)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return lbp_hist / np.sum(lbp_hist)  


def extract_frequency_features(image):
    gray_image = rgb2gray(image)
    f_transform = fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    return np.mean(magnitude_spectrum), np.std(magnitude_spectrum)

# Edge Detection with Sobel Filter
def extract_edge_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  
    edge_magnitude = np.hypot(sobelx, sobely)
    return np.mean(edge_magnitude), np.std(edge_magnitude)

def extract_hog_features(image):
    gray_image = rgb2gray(image)
    hog_features, _ = hog(gray_image, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)
    return hog_features

# Combine all extracted features 
def extract_all_features(image_path):
    image = load_image(image_path)
    if image is None:
        return None
    features = []
    texture_features = extract_texture_features(image)
    features.extend(texture_features)

    frequency_mean, frequency_std = extract_frequency_features(image)
    features.extend([frequency_mean, frequency_std])

    edge_mean, edge_std = extract_edge_features(image)
    features.extend([edge_mean, edge_std])
    
    hog_features = extract_hog_features(image)
    features.extend(hog_features)

    return np.array(features)

# Function to process all images in the specified directory and extract features
def extract_features_from_folder(folder_path):
    data = []
    labels = {"AI": 0, "real": 1}  

    for label_name, label in labels.items():
        label_folder = os.path.join(folder_path, label_name)
        if not os.path.isdir(label_folder):
            print(f"Warning: Folder {label_folder} does not exist.")
            continue

        for filename in os.listdir(label_folder):
            file_path = os.path.join(label_folder, filename)
            if filename.endswith((".jpg", ".jpeg", ".png")):
                features = extract_all_features(file_path)
                if features is None:
                    continue
                data.append((features, label))
                print(f"Processed {filename} from {label_name} folder.")

    return data

# Function to save features and labels to a .npz file
def save_features(data, save_path):
    features = np.array([item[0] for item in data])
    labels = np.array([item[1] for item in data])
    np.savez_compressed(save_path, features=features, labels=labels)
    print(f"Features saved to {save_path}")


folder_path = "/home/binit/classifier/data/feature_extraction"
data = extract_features_from_folder(folder_path)
save_path = "/home/binit/classifier/data/extracted_features.npz"
save_features(data, save_path)
