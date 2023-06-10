import time
import cv2

import numpy as np
import tensorflow as tf
from keras.metrics import MeanIoU, Precision, Recall
from keras.models import load_model

from potholes_detection.utils.constants import IM_HEIGHT, IM_WIDTH
from utils.read_data import read_images
from utils.augmentation import zip_generator, opening2d, closing2d

print("Reading test images...")

x_generator, y_generator = read_images(rgb_path="data/potholes_on_road/validation/images/",
                                       label_path="data/potholes_on_road/validation/masks/",
                                       im_height=IM_HEIGHT,
                                       im_width=IM_WIDTH)

test_generator = zip_generator(x_generator=x_generator, y_generator=y_generator)

# load the best model
model = load_model("saved_models/model.h5", compile=False)

test_iou_metric = MeanIoU(num_classes=2)
test_prec_metric = Precision()
test_rec_metric = Recall()


@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    y_pred = tf.where(y_pred >= 0.4, 1.0, 0.0)

    # # morphological operations
    # kernel = np.ones((5, 5, 1), np.float32)
    # # remove background noise
    # y_pred = opening2d(input=y_pred, kernel=kernel)
    # # remove foreground noise
    # y_pred = closing2d(input=y_pred, kernel=kernel)

    test_iou_metric.update_state(y, y_pred)
    test_prec_metric.update_state(y, y_pred)
    test_rec_metric.update_state(y, y_pred)


print("Running evaluation...")
start_time = time.time()
for x_batch_val, y_batch_val in test_generator:
    test_step(x_batch_val, y_batch_val)

test_iou = test_iou_metric.result()
test_prec = test_prec_metric.result()
test_rec = test_rec_metric.result()
test_iou_metric.reset_states()
test_prec_metric.reset_states()
test_rec_metric.reset_states()
print("Test meanIoU: %.4f" % (float(test_iou),))
print("Test precision: %.4f" % (float(test_prec),))
print("Test recall: %.4f" % (float(test_rec),))
print("Time taken: %.2fs" % (time.time() - start_time))
