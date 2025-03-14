import cv2
import numpy as np
from matplotlib import pyplot as plt

class AdaptiveMedianFilter:
    def NR_AMF(self, spath, dpath):
        # Load the noisy image
        image = cv2.imread(spath)

        max_window_size = 7
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        padded_image = np.pad(image, max_window_size // 2, mode='edge')
        output_image = np.copy(image)

        rows, cols = image.shape

        for i in range(rows):
            for j in range(cols):
                # Define the size of the window
                window_size = 3
                while window_size <= max_window_size:
                    # Extract the window
                    window = padded_image[i:i + window_size, j:j + window_size].flatten()
                    # Compute the median and the minimum and maximum values in the window
                    median = np.median(window)
                    min_val = np.min(window)
                    max_val = np.max(window)

                    # Apply the adaptive median filter logic
                    if min_val < median < max_val:
                        if image[i, j] < min_val or image[i, j] > max_val:
                            output_image[i, j] = median
                        else:
                            output_image[i, j] = image[i, j]
                        break
                    else:
                        window_size += 2  # Increase the window size

        plt.imsave(dpath, output_image)
