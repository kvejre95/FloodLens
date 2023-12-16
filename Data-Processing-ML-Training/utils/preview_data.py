'''
Code to preview/visualize FloodNet samples
'''

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
print(os.getcwd())
img_path = './data/train/train-org-img/6549.jpg'
image = Image.open(img_path)

lab_path = './data/train/train-label-img/6549_lab.png'
lab = Image.open(lab_path)

# Convert labels to numpy array for easier manipulation
lab_np = np.asarray(lab)

# Set water and flooding labels to 1 and others to 0
flooding_labels = [1, 3, 5]
binary_labels = np.isin(lab_np, flooding_labels).astype(np.uint8)

# Convert the numpy array back to an Image object
binary_labels_img = Image.fromarray(binary_labels*255)  

# Downscale the image and labels to 128x128
small_image = image.resize((128, 128))
small_lab = binary_labels_img.resize((128, 128))

# Display the downscaled image and labels side by side
fig, axes = plt.subplots(1, 2, figsize=(10,5))

# Display the downscaled image
axes[0].imshow(small_image)
axes[0].axis('off')
axes[0].set_title('Downscaled Image')

# Display the binary labels
axes[1].imshow(small_lab, cmap='gray')  
axes[1].axis('off')
axes[1].set_title('Binary Labels')

plt.tight_layout()
plt.show()



