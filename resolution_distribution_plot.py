import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Folder path containing the images
folder_path = 'data/post_processing'

# List to store the resolutions
resolutions = []

# Image filename extensions
extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG',
              '.gif', '.GIF', '.tif', '.tiff', '.TIF', '.TIFF',
              '.bmp', '.BMP', '.ico', '.ICO', '.eps', '.EPS']

# Iterate over each image in the folder
for filename in os.listdir(folder_path):
    # Add any other image formats if needed
    if filename.endswith(tuple(extensions)):
        image_path = os.path.join(folder_path, filename)
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Get the resolution
                width, height = img.size
                resolution = (width, height)
                resolutions.append(resolution)
        except (IOError, OSError):
            print(f"Error opening image: {filename}")

# Get min width and height and standard deviation for width and height
min_width = np.min([res[0] for res in resolutions])
min_height = np.min([res[1] for res in resolutions])
std_width = np.std([res[0] for res in resolutions])
std_height = np.std([res[1] for res in resolutions])
print(f"Minimum Width: {min_width}")
print(f"Minimum Height: {min_height}")
print(f"Standard Deviation of Width: {std_width}")
print(f"Standard Deviation of Height: {std_height}")

# Get max width and height
max_width = np.max([res[0] for res in resolutions])
max_height = np.max([res[1] for res in resolutions])
print(f"Maximum Width: {max_width}")
print(f"Maximum Height: {max_height}")


# Scatter plot of the resolutions
# Add opacity to the points for better visualization
plt.scatter(*zip(*resolutions), alpha=0.4, s=9)
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Image Resolutions')
plt.show()
