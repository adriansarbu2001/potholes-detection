from enum import Enum

from keras.layers import Activation, Dense, GlobalAveragePooling2D, Permute
from keras.layers.core import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.merging import multiply, dot, add


def position_attention_module(input):  # H x W x C
    x1 = Conv2D(filters=1, kernel_size=(1, 1), padding="same")(input)  # H x W x 1
    x2 = Activation("sigmoid")(x1)
    x3 = multiply([input, x2])  # H x W x C
    return x3


def channel_attention_module(input):  # H x W x C
    x1 = GlobalAveragePooling2D()(input)  # 1 x 1 x C
    x2 = Dense(input.shape[3] / 16, activation="relu")(x1)  # 1 x 1 x (C / 16)
    x3 = Dense(input.shape[3])(x2)  # 1 x 1 x C
    x4 = Activation("sigmoid")(x3)
    x5 = multiply([input, x4])  # H x W x C
    return x5


def dual_attention_module(input):  # H x W x C
    x1 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation="relu", padding="same")(input)  # H x W x C
    x2 = _channel_matrix_operation(x1)  # C x C
    x3 = Permute((2, 1))(x2)  # C x C
    x4 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(x1)  # (H * W) x C
    x5 = dot([x4, x3], axes=(2, 1))  # (H * W) x C
    x6 = Reshape((input.shape[1], input.shape[2], input.shape[3]))(x5)  # H x W x C
    x7 = add([x1, x6])  # H x W x C

    y1 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation="relu", padding="same")(input)  # H x W x C
    y2 = _spatial_matrix_operation(y1)  # (H * W) x (H * W)
    y3 = Permute((2, 1))(y2)  # (H * W) x (H * W)
    y4 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(y1)  # (H * W) x C
    y5 = dot([y3, y4], axes=(2, 1))  # (H * W) x C
    y6 = Reshape((input.shape[1], input.shape[2], input.shape[3]))(y5)  # H x W x C
    y7 = add([y1, y6])  # H x W x C

    rez = add([x7, y7])  # H x W x C
    return rez


def _channel_matrix_operation(input):  # H x W x C
    x1 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(input)  # (H * W) x C
    x2 = Permute((2, 1))(x1)  # C x (H * W)
    x3 = dot([x2, x1], axes=(2, 1))  # C x C
    x4 = Activation("softmax")(x3)
    return x4


def _spatial_matrix_operation(input):  # H x W x C
    x1 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation="relu", padding="same")(input)  # H x W x C
    x2 = Conv2D(filters=input.shape[3], kernel_size=(3, 3), activation="relu", padding="same")(input)  # H x W x C
    x1 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(x1)  # (H * W) x C
    x2 = Reshape((input.shape[1] * input.shape[2], input.shape[3]))(x2)  # (H * W) x C
    x2 = Permute((2, 1))(x2)  # C x (H * W)
    x3 = dot([x1, x2], axes=(2, 1))  # (H * W) x (H * W)
    x4 = Activation("softmax")(x3)
    return x4


class AM(Enum):
    NONE = (lambda x: x)
    POSITION = position_attention_module
    CHANNEL = channel_attention_module
    DUAL = dual_attention_module
