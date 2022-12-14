import base64
import io
import math
import sys
import cv2

import numpy as np
from PIL.Image import Image
from matplotlib import pyplot as plt

from skimage.transform import resize
from skimage import data, filters

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merging import concatenate
from keras.optimizers import Adam
from keras.utils import img_to_array
from flask import Flask, request, jsonify, Response, make_response

# Set some parameters
im_width = 400
im_height = 400
border = 5


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

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    return Model(inputs=[input_img], outputs=[outputs])


input_img = Input((im_height, im_width, 3), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# model.summary()

# load the best model
model.load_weights('model-tgs-salt.h5')

app = Flask(__name__)


@app.route('/potholes-detection', methods=['POST'])
def get_contour():
    try:
        r = request
        # convert string of image data to uint8
        nparr = np.frombuffer(r.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # plt.imshow(img)
        # plt.show()

        x_img = img
        x_img = resize(x_img, (400, 400, 3), mode='constant', preserve_range=True)
        x_img = x_img / 255.0

        res = model.predict(np.array([x_img]))[0]
        res = np.where(res > 0.75, 1, res)
        res = np.where(res < 0.75, 0, res)
        # edges = filters.sobel(res)

        norm_res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        norm_res = cv2.cvtColor(norm_res, cv2.COLOR_GRAY2BGR)

        red_channel = norm_res[:, :, 2]
        # create empty image with same shape as that of src image
        red_img = np.zeros((norm_res.shape[0], norm_res.shape[1], norm_res.shape[2] + 1))
        # assign the red channel of src to empty image
        red_img[:, :, 2] = red_channel
        red_img[:, :, 3] = red_channel // 3

        # edges = cv2.normalize(edges, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # has_mask = x_img.max() > 0
        #
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
        # # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15))
        #
        # ax1.imshow(x_img)
        # if has_mask:
        #     # draw a boundary(contour) in the original image separating pothole and background areas
        #     ax1.contour(res[0].squeeze(), colors='k', linewidths=5, levels=[0.5])
        # ax1.set_title('Test image')
        #
        # ax2.imshow(res[0].squeeze(), cmap='gray', interpolation='bilinear')
        # ax2.set_title('Predicted')
        # # ax3.imshow(label.squeeze(), cmap='gray', interpolation='bilinear')
        # # ax3.set_title('Real label')

        retval, buffer = cv2.imencode('.png', red_img)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/png'
        return response

    except Exception as err:
        print(err)
        return '', 500


@app.after_request
def after_request(response):
    print("log: setting cors", file=sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    app.run()
