import os
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, GlobalAveragePooling2D, Permute
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merging import concatenate, multiply, dot, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img

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

ids = next(os.walk("data/training/rgb"))[2]  # list of names all images in the given path
print("No. of images = ", len(ids))

X_train = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
y_train = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

# tqdm is used to display the progress bar
for n, id_ in zip(range(len(ids)), ids):
    # Load images
    img = load_img("data/training/rgb/" + id_, color_mode='rgb')
    x_img = img_to_array(img)
    x_img = resize(x_img, (400, 400, 3), mode='constant', preserve_range=True)
    # Load masks
    mask = img_to_array(load_img("data/training/label/" + id_, color_mode='grayscale'))
    mask = resize(mask, (400, 400, 1), mode='constant', preserve_range=True)
    # Save images
    X_train[n] = x_img / 255.0
    y_train[n] = mask / 255.0

ids = next(os.walk("data/validation/rgb"))[2]  # list of names all images in the given path
print("No. of images = ", len(ids))

X_valid = np.zeros((len(ids), im_height, im_width, 3), dtype=np.float32)
y_valid = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

# tqdm is used to display the progress bar
for n, id_ in zip(range(len(ids)), ids):
    # Load images
    img = load_img("data/validation/rgb/" + id_, color_mode='rgb')
    x_img = img_to_array(img)
    x_img = resize(x_img, (400, 400, 3), mode='constant', preserve_range=True)
    # Load masks
    mask = img_to_array(load_img("data/validation/label/" + id_, color_mode='grayscale'))
    mask = resize(mask, (400, 400, 1), mode='constant', preserve_range=True)
    # Save images
    X_valid[n] = x_img / 255.0
    y_valid[n] = mask / 255.0

# Visualize any random image along with the mask
ix = random.randint(0, len(X_train))
has_mask = y_train[ix].max() > 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

print(np.array(X_train[ix]).shape)

ax1.imshow(X_train[ix])
if has_mask:
    # draw a boundary(contour) in the original image separating pothole and background areas
    ax1.contour(y_train[ix].squeeze(), colors='k', linewidths=5, levels=[0.5])
ax1.set_title('RGB')

ax2.imshow(y_train[ix].squeeze(), cmap='gray', interpolation='bilinear')
ax2.set_title('Segmentation')
plt.show()


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def channel_matrix_operation(input):  # H x W x C
    x1 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(input)   # (H * W) x C
    x2 = Permute((2, 1))(x1)  # C x (H * W)
    x3 = dot([x2, x1], axes=(2, 1))  # C x C
    x4 = Activation('softmax')(x3)
    return x4


def spatial_matrix_operation(input, batchnorm=True):  # H x W x C
    x1 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation='relu', padding='same')(input)  # H x W x C
    if batchnorm:
        x1 = BatchNormalization()(x1)
    x2 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation='relu', padding='same')(input)  # H x W x C
    if batchnorm:
        x2 = BatchNormalization()(x2)
    x1 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(x1)  # (H * W) x C
    x2 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(x2)  # (H * W) x C
    x2 = Permute((2, 1))(x2)  # C x (H * W)
    x3 = dot([x1, x2], axes=(2, 1))  # (H * W) x (H * W)
    x4 = Activation('softmax')(x3)
    return x4


def position_attention_module(input):  # H x W x C
    x1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(input)  # H x W x 1
    x2 = Activation('sigmoid')(x1)
    x3 = multiply([input, x2])  # H x W x C
    return x3


def channel_attention_module(input):  # H x W x C
    x1 = GlobalAveragePooling2D()(input)  # 1 x 1 x C
    x2 = Dense(input.shape[3] / 16, activation='relu')(x1)  # 1 x 1 x (C / 16)
    x3 = Dense(input.shape[3])(x2)  # 1 x 1 x C
    x4 = Activation('sigmoid')(x3)
    x5 = multiply([input, x4])  # H x W x C
    return x5


def dual_attention_module(input, batchnorm=True):  # H x W x C
    x1 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation='relu', padding='same')(input)  # H x W x C
    x2 = channel_matrix_operation(x1)  # C x C
    x3 = Permute((2, 1))(x2)  # C x C
    x4 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(x1)  # (H * W) x C
    x5 = dot([x4, x3], axes=(2, 1))  # (H * W) x C
    x6 = Reshape((input.shape[1], input.shape[2], input.shape[3]))(x5)  # H x W x C
    x7 = add([x1, x6])  # H x W x C

    y1 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation='relu', padding='same')(input)  # H x W x C
    y2 = spatial_matrix_operation(y1, batchnorm)  # (H * W) x (H * W)
    y3 = Permute((2, 1))(y2)  # (H * W) x (H * W)
    y4 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(y1)  # (H * W) x C
    y5 = dot([y3, y4], axes=(2, 1))  # (H * W) x C
    y6 = Reshape((input.shape[1], input.shape[2], input.shape[3]))(y5)  # H x W x C
    y7 = add([y1, y6])  # H x W x C

    rez = add([x7, y7])  # H x W x C
    return rez


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    am5 = dual_attention_module(c5, batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(am5)
    am4 = channel_attention_module(c4)
    u6 = concatenate([u6, am4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    am3 = channel_attention_module(c3)
    u7 = concatenate([u7, am3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    am2 = channel_attention_module(c2)
    u8 = concatenate([u8, am2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    am1 = position_attention_module(c1)
    u9 = concatenate([u9, am1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return Model(inputs=[input_img], outputs=[outputs])


input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
]

results = model.fit(X_train, y_train, batch_size=8, epochs=50, callbacks=callbacks, validation_data=(X_valid, y_valid))

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()
plt.show()
