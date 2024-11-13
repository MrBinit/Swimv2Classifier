import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray
from scipy.fft import fft2
from PIL import Image

# Function to load image and convert to grayscale
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return np.array(image)

# Texture Analysis with Local Binary Patterns (LBP)
def extract_texture_features(image):
    gray_image = rgb2gray(image)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return lbp_hist / np.sum(lbp_hist)  # Normalize histogram

# Frequency Domain Analysis using Fourier Transform
def extract_frequency_features(image):
    gray_image = rgb2gray(image)
    f_transform = fft2(gray_image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    return np.mean(magnitude_spectrum), np.std(magnitude_spectrum)

# Edge Detection with Sobel Filter
def extract_edge_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
    edge_magnitude = np.hypot(sobelx, sobely)
    return np.mean(edge_magnitude), np.std(edge_magnitude)

# Histogram of Oriented Gradients (HOG) for structural information
def extract_hog_features(image):
    gray_image = rgb2gray(image)
    hog_features, _ = hog(gray_image, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True)
    return hog_features

# Combine all extracted features into a single vector
def extract_all_features(image_path):
    image = load_image(image_path)
    features = []
    
    # Extract texture features
    texture_features = extract_texture_features(image)
    features.extend(texture_features)
    
    # Extract frequency features
    frequency_mean, frequency_std = extract_frequency_features(image)
    features.extend([frequency_mean, frequency_std])
    
    # Extract edge features
    edge_mean, edge_std = extract_edge_features(image)
    features.extend([edge_mean, edge_std])
    
    # Extract HOG features
    hog_features = extract_hog_features(image)
    features.extend(hog_features)

    return np.array(features)

# Example usage
image_path = "/home/binit/classifier/data/raw/AI/download (22) copy 2.jpeg"
features = extract_all_features(image_path)
print("Extracted features:", features)
