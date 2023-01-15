import os
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize
from sklearn.model_selection import train_test_split

from keras.models import load_model
from keras.utils import img_to_array, load_img

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    except RuntimeError as e:
        print(e)
"""

plt.style.use("ggplot")

# Set some parameters
im_width = 400
im_height = 400
border = 5

ids = next(os.walk("data/testing/rgb"))[2]  # list of names all images in the given path
print("No. of images = ", len(ids))

X = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

for n, id_ in zip(range(len(ids)), ids):
    # Load images
    img = load_img("data/training/rgb/" + id_, color_mode='rgb')
    x_img = img_to_array(img)
    x_img = resize(x_img, (400, 400, 3), mode='constant', preserve_range=True)
    # Load masks
    mask = img_to_array(load_img("data/training/label/" + id_, color_mode='grayscale'))
    mask = resize(mask, (400, 400, 1), mode='constant', preserve_range=True)
    # Save images
    X[n] = x_img / 255.0
    y[n] = mask / 255.0

# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)

# Visualize any random image along with the mask
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

ax1.imshow(X_train[ix])
if has_mask:
    # draw a boundary(contour) in the original image separating pothole and background areas
    ax1.contour(y_train[ix].squeeze(), colors='k', linewidths=5, levels=[0.5])
ax1.set_title('RGB')

ax2.imshow(y_train[ix].squeeze(), cmap='gray', interpolation='bilinear')
ax2.set_title('Segmentation')
plt.show()

# load the best model
model = load_model('model.h5')

# model.summary()

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(X_valid, y_valid, verbose=1)
