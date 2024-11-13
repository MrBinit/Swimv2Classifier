# import numpy as np
# from skimage.feature import local_binary_pattern
# from skimage.color import rgb2gray
# from PIL import Image

# # Function to load image and convert to grayscale
# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return np.array(image)

# # Texture Analysis with Local Binary Patterns (LBP)
# def extract_lbp_features(image):
#     gray_image = rgb2gray(image)
#     gray_image = (gray_image * 255).astype(np.uint8)
    
#     # Apply Local Binary Pattern
#     lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    
#     # Calculate histogram of LBP
#     lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    
#     # Normalize histogram
#     lbp_hist = lbp_hist / np.sum(lbp_hist)
    
#     return lbp_hist

# # Example usage
# image_path = "data/augmented/real/9U9A9587.JPG" 
# image = load_image(image_path)
# lbp_features = extract_lbp_features(image)

# print("Extracted LBP features (histogram):", lbp_features)



# image_path = "data/augmented/real/9U9A9587.JPG" 
# /home/binit/classifier/data/raw/AI/download (22) copy 2.jpeg


# import numpy as np
# from scipy.fft import fft2, fftshift
# from PIL import Image
# from skimage.color import rgb2gray

# # Function to load and convert image to grayscale
# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return np.array(image)

# # Frequency Analysis using Fourier Transform
# def extract_frequency_features(image):
#     # Convert the image to grayscale
#     gray_image = rgb2gray(image)
    
#     # Apply 2D Fourier Transform
#     f_transform = fft2(gray_image)
    
#     # Shift the zero frequency component to the center
#     f_shift = fftshift(f_transform)
    
#     # Compute the magnitude spectrum
#     magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
#     # Calculate mean and standard deviation of the magnitude spectrum
#     mean_frequency = np.mean(magnitude_spectrum)
#     std_frequency = np.std(magnitude_spectrum)
    
#     return mean_frequency, std_frequency

# # Example usage
# image_path = "data/augmented/real/9U9A9587.JPG"
# image = load_image(image_path)
# mean_freq, std_freq = extract_frequency_features(image)

# print(f"Mean Frequency: {mean_freq:.4f}")
# print(f"Standard Deviation of Frequency: {std_freq:.4f}")

# import cv2
# import numpy as np
# from PIL import Image

# # Function to load image and convert to grayscale
# def load_image(image_path):
#     image = Image.open(image_path).convert("RGB")
#     return np.array(image)

# # Edge Detection with Sobel Filter
# def extract_sobel_edge_features(image):
#     # Convert image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
#     # Apply Sobel filter to detect edges in x and y directions
#     sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal edges
#     sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Vertical edges
    
#     # Calculate the magnitude of the gradient
#     edge_magnitude = np.hypot(sobelx, sobely)
    
#     # Calculate the mean and standard deviation of edge magnitudes
#     edge_mean = np.mean(edge_magnitude)
#     edge_std = np.std(edge_magnitude)
    
#     return edge_mean, edge_std

# # Example usage
# image_path = "data/augmented/real/9U9A9587.JPG"
# image = load_image(image_path)
# edge_mean, edge_std = extract_sobel_edge_features(image)

# print("Sobel Edge Mean:", edge_mean)
# print("Sobel Edge Standard Deviation:", edge_std)


import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt

# Function to load and convert the image to grayscale
def load_image(image_path):
    image = io.imread(image_path)
    return rgb2gray(image)  # Convert to grayscale, as HOG works best on single channel images

# Function to extract HOG features
def extract_hog_features(image_path, visualize=False):
    # Load and preprocess image
    gray_image = load_image(image_path)

    # Compute HOG features
    hog_features, hog_image = hog(
        gray_image,
        pixels_per_cell=(16, 16),     # Size of cell for HOG computation
        cells_per_block=(2, 2),       # Number of cells per block
        block_norm='L2-Hys',          # Block normalization
        visualize=True,               # Return HOG image for visualization
        feature_vector=True           # Return features as a single vector
    )
    
    # If visualization is enabled, display the HOG image
    if visualize:
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.title("Grayscale Image")
        plt.imshow(gray_image, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("HOG Image")
        plt.imshow(hog_image, cmap="gray")
        plt.axis("off")
        plt.show()

    return hog_features

# Example usage
image_path = "/home/binit/classifier/data/raw/AI/download (22) copy 2.jpeg"
hog_features = extract_hog_features(image_path, visualize=True)
print("Extracted HOG features:", hog_features)
print("Number of HOG features:", len(hog_features))
