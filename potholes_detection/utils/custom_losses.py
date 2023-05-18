import tensorflow as tf
import keras.losses as losses


def weighted_binary_crossentropy(weight_foreground, weight_background):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        b_ce = losses.BinaryCrossentropy()(y_true, y_pred)
        weight_vector = y_true * weight_foreground + (1 - y_true) * weight_background
        weighted_b_ce = weight_vector * b_ce
        return tf.reduce_mean(weighted_b_ce)

    return loss


def focal_loss(gamma):
    _EPSILON = tf.keras.backend.epsilon()

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Predicted probabilities for the negative class
        negative_pred = 1 - y_pred

        # For numerical stability (so we don't inadvertently take the log of 0)
        y_pred = tf.math.maximum(y_pred, _EPSILON)
        negative_pred = tf.math.maximum(negative_pred, _EPSILON)

        # Loss for the positive examples
        pos_loss = -(negative_pred ** gamma) * tf.math.log(y_pred)

        # Loss for the negative examples
        neg_loss = -(y_pred ** gamma) * tf.math.log(negative_pred)

        loss = y_true * pos_loss + (1 - y_true) * neg_loss
        return tf.reduce_mean(loss)

    return loss_fn


def weighted_focal_loss(weight_foreground, weight_background, gamma):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        b_ce = focal_loss(gamma)(y_true, y_pred)
        weight_vector = y_true * weight_foreground + (1 - y_true) * weight_background
        weighted_b_ce = weight_vector * b_ce
        return tf.reduce_mean(weighted_b_ce)

    return loss
