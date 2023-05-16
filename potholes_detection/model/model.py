from keras.losses import BinaryCrossentropy
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merging import concatenate
from keras.optimizers import Adam

from potholes_detection.model.attention_modules import AM
from potholes_detection.utils.constants import IM_HEIGHT, IM_WIDTH, POTHOLE_WEIGHT, BACKGROUND_WEIGHT
from potholes_detection.utils.custom_losses import weighted_binary_crossentropy


def _conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def get_unet(am_scheme=(AM.NONE, AM.NONE, AM.NONE, AM.NONE, AM.NONE), n_filters=16, dropout=0.1, batchnorm=True):
    inp = Input((IM_HEIGHT, IM_WIDTH, 3), name="img")

    # Contracting Path
    c1 = _conv2d_block(inp, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = _conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = _conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = _conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = _conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    am5 = am_scheme[4](c5)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding="same")(am5)
    am4 = am_scheme[3](c4)
    u6 = concatenate([u6, am4])
    u6 = Dropout(dropout)(u6)
    c6 = _conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding="same")(c6)
    am3 = am_scheme[2](c3)
    u7 = concatenate([u7, am3])
    u7 = Dropout(dropout)(u7)
    c7 = _conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding="same")(c7)
    am2 = am_scheme[1](c2)
    u8 = concatenate([u8, am2])
    u8 = Dropout(dropout)(u8)
    c8 = _conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding="same")(c8)
    am1 = am_scheme[0](c1)
    u9 = concatenate([u9, am1])
    u9 = Dropout(dropout)(u9)
    c9 = _conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    return Model(inputs=[inp], outputs=[outputs])


def get_optimizer():
    return Adam(learning_rate=1e-4)


def get_loss_fn():
    # return BinaryCrossentropy()
    return weighted_binary_crossentropy(POTHOLE_WEIGHT, BACKGROUND_WEIGHT)
