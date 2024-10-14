import os
from PIL import Image
import numpy as np

# Load image
image_folder = '/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint/altitude2.0_offsetX0_offsetY0_scale0.125/composite/images'
image_path = '/mnt/hdd/code/human_data_generation/xrfeitoria/output/waypoint/altitude2.0_offsetX0_offsetY0_scale0.125/composite/images/01037.jpg'  # Replace with your image file path

for file in os.listdir(image_folder):
    print(file)
    image_path = os.path.join(image_folder, file)
    img = Image.open(image_path)

    # Convert image to NumPy array
    img_array = np.array(img)

    # min max pixel value
    min_pixel_value = np.min(img_array)
    max_pixel_value = np.max(img_array)

    if 0 > min_pixel_value or 255 < max_pixel_value:
        print("Image path:", image_path)
        print("Min pixel value:", min_pixel_value)
        print("Max pixel value:", max_pixel_value)


# # Check pixel values
# print("Image shape:", img_array.shape)  # Shape of the image (height, width, channels)
# #min max pixel value
# print("Min pixel value:", np.min(img_array))
# print("Max pixel value:", np.max(img_array))