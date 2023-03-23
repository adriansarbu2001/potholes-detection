import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from keras.utils import img_to_array, load_img

plt.style.use("ggplot")

# Set some parameters
im_width = 400
im_height = 400
border = 5

ids = next(os.walk("data/training/rgb"))[2]  # list of names all images in the given path
print("No. of images = ", len(ids))

y_train = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

ratios = []
for n, id_ in zip(range(len(ids)), ids):
    # Load masks
    mask = img_to_array(load_img("data/training/label/" + id_, color_mode='grayscale'))
    mask = resize(mask, (400, 400, 1), mode='constant', preserve_range=True)
    # Save images
    y_train[n] = mask / 255.0

    unique, counts = np.unique(mask, return_counts=True)
    ratios.append(counts[1] / (counts[0] + counts[1]))

print(sum(ratios) / len(ratios))
plt.xlabel("Pothole area / image area", fontsize=16)
plt.ylabel("No. images", fontsize=16)
plt.hist(ratios, bins=15)
plt.show()
