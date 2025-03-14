import cv2
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt

class Handcraft_Features:
    def extract_features(self, img_path):
        from skimage.feature import greycomatrix, greycoprops
        # Load the image
        image_path = img_path  # Replace with the path to your image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 1. Edge Detection using Canny
        edges = cv2.Canny(image, 100, 200)

        # 2. Corner Detection using Harris
        dst = cv2.cornerHarris(image, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)  # Dilate to mark corners
        corners = np.zeros_like(image)
        corners[dst > 0.01 * dst.max()] = 255  # Thresholding to highlight corners

        # 3. Texture Features using Grey Level Co-occurrence Matrix (GLCM)
        # Ensure the image has the correct format for GLCM
        glcm = greycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = greycoprops(glcm, 'contrast')[0, 0]
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
        energy = greycoprops(glcm, 'energy')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]

        features = []
        features.append(glcm)
        features.append(contrast)
        features.append(dissimilarity)
        features.append(homogeneity)
        features.append(energy)
        features.append(correlation)

        return features