import time

import tensorflow as tf
from keras.metrics import MeanIoU, Accuracy
from keras.models import load_model
from utils.read_data import read_images
from utils.augmentation import zip_generator

print("Reading test images...")
x_generator, y_generator = read_images(rgb_path="data/testing/rgb/", label_path="data/testing/label/")
test_generator = zip_generator(x_generator=x_generator, y_generator=y_generator)

# load the best model
model = load_model("custom_trained_model_v3.h5", compile=False)

test_iou_metric = MeanIoU(num_classes=2)
test_acc_metric = Accuracy()


@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    y_pred = tf.where(y_pred >= 0.5, 1.0, 0.0)
    test_iou_metric.update_state(y, y_pred)
    test_acc_metric.update_state(y, y_pred)


print("Running evaluation...")
start_time = time.time()
for x_batch_val, y_batch_val in test_generator:
    test_step(x_batch_val, y_batch_val)

test_iou = test_iou_metric.result()
test_acc = test_acc_metric.result()
test_iou_metric.reset_states()
test_acc_metric.reset_states()
print("Test meanIoU: %.4f" % (float(test_iou),))
print("Test accuracy: %.4f" % (float(test_acc),))
print("Time taken: %.2fs" % (time.time() - start_time))
