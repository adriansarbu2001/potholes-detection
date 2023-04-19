import os

import numpy as np
from skimage.transform import resize

from keras.utils import img_to_array, load_img
from potholes_detection.utils.constants import IM_HEIGHT, IM_WIDTH


def read_images(rgb_path, label_path):
    filenames = next(os.walk(rgb_path))[2]
    # print("No. of images = ", len(ids))

    x = np.zeros((len(filenames), IM_HEIGHT, IM_WIDTH, 3), dtype=np.float32)
    y = np.zeros((len(filenames), IM_HEIGHT, IM_WIDTH, 1), dtype=np.uint8)

    for index, filename in enumerate(filenames):
        # Load images
        img = load_img(rgb_path + filename, color_mode="rgb")
        x_img = img_to_array(img)
        x_img = resize(x_img, (IM_HEIGHT, IM_WIDTH, 3), mode="constant", preserve_range=True)

        # Load masks
        mask = img_to_array(load_img(label_path + filename, color_mode="grayscale"))
        mask = resize(mask, (IM_HEIGHT, IM_WIDTH, 1), mode="constant", preserve_range=True)

        # Save images
        x[index] = x_img / 255
        y[index] = mask / 255

    return x, y
