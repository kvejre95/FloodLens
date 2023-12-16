'''
Code for preprocessing Sentinel water bodies dataset
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.utils import shuffle

image_path = "./Water Bodies Dataset/Images/*.jpg"
mask_path = "./Water Bodies Dataset/Masks/*.jpg"

image_paths = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_paths = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])

def resize_mask(mask, size):
    # Convert the mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Resize the mask to the specified size
    resized_mask = cv2.resize(mask_gray, size, interpolation=cv2.INTER_NEAREST)
    resized_mask = np.expand_dims(resized_mask, axis=2)
    return resized_mask

def resize_image(image, size):
    # Resize the image to the specified size
    resized_image = cv2.resize(image, size)
    return resized_image

# Define the desired size
target_size = (256,256)

image_list = []
mask_list = []

for image_path, mask_path in zip(image_paths, mask_paths):
    # Load the image and mask
    image = plt.imread(image_path).astype(np.float32) / 255.
    mask = plt.imread(mask_path).astype(np.float32) / 255.

    # Resize the image and mask
    resized_image = resize_image(image, target_size)
    resized_mask = resize_mask(mask, target_size)
    resized_image = np.moveaxis(resized_image, -1, 0)
    resized_mask = np.moveaxis(resized_mask, -1, 0)

    image_list.append(resized_image)
    mask_list.append(resized_mask)

# Convert the image and mask lists to arrays
image_array = np.array(image_list)
mask_array = np.array(mask_list)
print(mask_array)
image_array, mask_array = shuffle(image_array, mask_array, random_state=42)

# Check the shapes of the resized image and mask arrays
print("Resized image array shape:", image_array.shape)
print("Resized mask array shape:", mask_array.shape)

np.save('./train_img.npy', image_array[:2820])
np.save('./test_img.npy', image_array[2820:])
np.save('./train_mask.npy', mask_array[:2820])
np.save('./test_mask.npy', mask_array[2820:])




 