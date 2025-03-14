import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

class Inception_V3:
    def extract_features(self, img_path):
        # Load the pre-trained Inception V3 model
        base_model = InceptionV3(weights='imagenet', include_top=False)
        # Function to preprocess and load an image
        def load_and_preprocess_image(img_path, target_size=(299, 299)):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)       # Preprocess the image
            return img_array

        # Function to extract features using Inception V3
        def extract_features(img_path):
            img_array = load_and_preprocess_image(img_path)
            features = base_model.predict(img_array)      # Extract features
            return features