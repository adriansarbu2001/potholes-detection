import tensorflow as tf
import keras.losses as losses


def weighted_binary_crossentropy(weight_foreground, weight_background):
    def loss(y_true, y_pred):
        b_ce = losses.BinaryCrossentropy()(y_true, y_pred)
        weight_vector = y_true * weight_foreground + (1 - y_true) * weight_background
        weighted_b_ce = weight_vector * b_ce
        return tf.reduce_mean(weighted_b_ce)

    return loss
