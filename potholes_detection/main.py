import sys
import cv2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

from keras.models import load_model
from flask import Flask, request, make_response
from custom_losses import weighted_binary_crossentropy

# Set some parameters
im_width = 400
im_height = 400
border = 5

# load the best model
# model = load_model('model-unet-3.h5', custom_objects={"loss": weighted_binary_crossentropy(0.92, 0.08)})
model = load_model('custom_trained_model_v5.h5', compile=False)

# model.summary()

app = Flask(__name__)


def opening2d(input, kernel):
    eroded = tf.nn.erosion2d(input, kernel, [1, 1, 1, 1], 'SAME', "NHWC", [1, 1, 1, 1])
    opened = tf.nn.dilation2d(eroded, kernel, [1, 1, 1, 1], 'SAME', "NHWC", [1, 1, 1, 1])
    return opened


def closing2d(input, kernel):
    dilated = tf.nn.dilation2d(input, kernel, [1, 1, 1, 1], 'SAME', "NHWC", [1, 1, 1, 1])
    closed = tf.nn.erosion2d(dilated, kernel, [1, 1, 1, 1], 'SAME', "NHWC", [1, 1, 1, 1])
    return closed


@app.route('/potholes-detection', methods=['POST'])
def get_contour():
    try:
        print("REQUEST RECEIVED")
        r = request
        # convert string of image data to uint8
        nparr = np.frombuffer(r.data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # plt.imshow(img)
        # plt.show()

        x_img = img
        x_img = resize(x_img, (400, 400, 3), mode='constant', preserve_range=True)
        x_img = x_img / 255.0

        res = model(np.array([x_img]), training=False)
        res = tf.where(res >= 0.5, 1.0, 0.0)

        # morphological operations
        kernel = np.ones((5, 5, 1), np.float32)
        # remove background noise
        res = opening2d(input=res, kernel=kernel)
        # remove foreground noise
        res = closing2d(input=res, kernel=kernel)

        res = res.numpy()[0]
        # edges = filters.sobel(res)

        norm_res = cv2.normalize(res, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        norm_res = cv2.cvtColor(norm_res, cv2.COLOR_GRAY2BGR)

        red_channel = norm_res[:, :, 2]
        # create empty image with same shape as that of src image
        red_img = np.zeros((norm_res.shape[0], norm_res.shape[1], norm_res.shape[2] + 1))
        # assign the red channel of src to empty image
        red_img[:, :, 2] = red_channel
        red_img[:, :, 3] = red_channel // 3

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
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Connection,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == "__main__":
    # app.run()
    # app.run(host='192.168.1.24', port=5000, debug=False)
    app.run(port=5000, debug=False)
