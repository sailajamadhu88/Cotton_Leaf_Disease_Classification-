import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def extract_features(self, img_path):
        # Load pre-trained VGG16 model without the top classification layers
        base_model = VGG16(weights='imagenet', include_top=False)

        # Define a feature extractor model (outputs the feature maps from the convolutional layers)
        feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

        # Function to preprocess an image for the VGG16 model
        def preprocess_image(image_path, target_size=(224, 224)):
            image = load_img(image_path, target_size=target_size)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = tf.keras.applications.vgg16.preprocess_input(image)
            return image

        # Function to visualize feature maps
        def visualize_feature_maps(feature_maps, num_columns=8):
            num_features = feature_maps.shape[-1]
            num_rows = (num_features + num_columns - 1) // num_columns

            plt.figure(figsize=(20, 20))
            for i in range(num_features):
                plt.subplot(num_rows, num_columns, i + 1)
                plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
                plt.axis('off')
            plt.show()
