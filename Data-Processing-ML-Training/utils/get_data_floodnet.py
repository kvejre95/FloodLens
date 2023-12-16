'''
Code for FloodNet dataset preprocessing
'''

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

mypath1 = './data/train/train-org-img'
files1 = [os.path.join(mypath1, f) for f in os.listdir(mypath1) if f.endswith(".jpg")]
mypath1_val = './data/val/val-org-img'
files1_val = [os.path.join(mypath1_val, f) for f in os.listdir(mypath1_val) if f.endswith(".jpg")]
mypath1_test = './data/test/test-org-img'
files1_test = [os.path.join(mypath1_test, f) for f in os.listdir(mypath1_test) if f.endswith(".jpg")]
files1 = files1 + files1_val + files1_test
files1.sort()

mypath2 = './data/train/train-label-img'
files2 = [os.path.join(mypath2, f) for f in os.listdir(mypath2) if f.endswith(".png")]
mypath2_val = './data/val/val-label-img'
files2_val = [os.path.join(mypath2_val, f) for f in os.listdir(mypath2_val) if f.endswith(".png")]
mypath2_test = './data/test/test-label-img'
files2_test = [os.path.join(mypath2_test, f) for f in os.listdir(mypath2_test) if f.endswith(".png")]
files2 = files2 + files2_val + files2_test
files2.sort()

print(len(files1))
print(len(files2))

def normalize_lab(lab):
    normalized = (lab - 1) / 9.0
    rescaled = 2 * normalized - 1
    return rescaled

imgs_orig = []
imgs = []
labs = []
for i in tqdm(range(len(files1))):
    img = Image.open(files1[i])
    img_orig = np.asarray(img.resize((64, 64)))
    imgs_orig.append(img_orig)

    sample = np.asarray(img.resize((64, 64)).convert("L"))
    sample = sample = (sample - np.amin(sample))/(np.amax(sample) - np.amin(sample))
    sample = 2*sample-1
    img = sample.reshape(1,64,64)

    lab = Image.open(files2[i])
    lab = np.asarray(lab.resize((64, 64)))
    lab = normalize_lab(lab)
    lab = lab.reshape(1,64,64)

    imgs.append(img)
    labs.append(lab)
imgs_orig = np.asarray(imgs_orig)
imgs = np.asarray(imgs)
labs = np.asarray(labs)
print('Done!')
print(imgs_orig.shape)
print(imgs.shape)
print(labs.shape)

print('Splitting and Saving')
train_imgs, test_imgs, train_labs, test_labs = train_test_split(imgs, labs, test_size=0.10, shuffle=True, random_state=42)

np.save('./train_images_original.npy', imgs_orig)
np.save('./train_images.npy', train_imgs)
np.save('./train_labels.npy', train_labs)
np.save('./test_images.npy', test_imgs)
np.save('./test_labels.npy', test_labs)