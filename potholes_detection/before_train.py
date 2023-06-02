import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from keras.utils import img_to_array, load_img
from utils.read_data import read_images
from utils.augmentation import zip_generator, zip_generator_with_augmentation

plt.style.use("ggplot")

x_generator, y_generator = read_images(rgb_path="data/potholes_on_road/training/images/",
                                                   label_path="data/potholes_on_road/training/masks/")

generator = zip_generator(x_generator, y_generator)
# generator = zip_generator(x_generator, y_generator)

ratios = []
for x_batch, y_batch in generator:
    for mask in y_batch:
        unique, counts = np.unique(mask, return_counts=True)
        ratios.append(counts[1] / (counts[0] + counts[1]))


print(sum(ratios) / len(ratios))
plt.xlabel("Pothole area / image area", fontsize=16)
plt.ylabel("No. images", fontsize=16)
plt.hist(ratios, bins=30)
plt.show()
