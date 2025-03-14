import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_se=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_se = use_se
        self.se = SEBlock(out_channels) if use_se else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_se:
            x = self.se(x)
        return x

class CNNSE_MIV3(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNSE_MIV3, self).__init__()
        self.layer1 = ConvBlock(3, 32, kernel_size=3, stride=1, padding=1, use_se=True)
        self.layer2 = ConvBlock(32, 64, kernel_size=3, stride=1, padding=1, use_se=True)
        self.layer3 = ConvBlock(64, 128, kernel_size=3, stride=1, padding=1, use_se=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)  # Assuming input images are 32x32

    def extract_features(self, img_path):
        # Function to preprocess and load an image
        def load_and_preprocess_image(img_path, target_size=(299, 299)):
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = preprocess_input(img_array)       # Preprocess the image
            return img_array

        def extract_features(img_path):
            img_array = load_and_preprocess_image(img_path)
            features = self.base_model.predict(img_array)      # Extract features
            return features
        def build_modified_inceptionv3(input_shape=(299, 299, 3), output_layer='mixed10'):
            """
            Builds a modified InceptionV3 model for feature extraction.

            Parameters:
            - input_shape: tuple, the shape of input images (default: (299, 299, 3)).
            - output_layer: str, the name of the layer to extract features from.

            Returns:
            - model: a Keras Model for feature extraction.
            """
            # Load the InceptionV3 model with ImageNet weights
            base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

            # Check if the specified layer exists in the model
            if output_layer not in [layer.name for layer in base_model.layers]:
                raise ValueError(
                    f"Layer '{output_layer}' not found in InceptionV3. Available layers: {[layer.name for layer in base_model.layers]}")

            # Create a new model that outputs features from the specified layer
            model = Model(inputs=base_model.input, outputs=base_model.get_layer(output_layer).output)

            return model

    def forward(self, x):
        x = self.layer1(x)
        x = self.pool(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x