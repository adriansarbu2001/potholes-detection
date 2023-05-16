import os

import tensorflow as tf
from potholes_detection.utils.constants import IM_HEIGHT, IM_WIDTH


def read_images(rgb_path, label_path):
    # Get a list of image filenames
    filenames = next(os.walk(rgb_path))[2]

    def load_and_preprocess_image(filename):
        # Load the image and resize it
        img = tf.io.read_file(rgb_path + filename)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.BICUBIC)

        # Load the label and resize it
        mask = tf.io.read_file(label_path + filename)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, (IM_HEIGHT, IM_WIDTH), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Normalize the image and label
        img = tf.cast(img, tf.float32) / 255.0
        mask = tf.cast(mask, tf.uint8) // 255

        return img, mask

    xy = tf.data.Dataset.from_tensor_slices(filenames).map(load_and_preprocess_image)

    x = xy.map(lambda x, y: x)
    y = xy.map(lambda x, y: y)

    return x, y
