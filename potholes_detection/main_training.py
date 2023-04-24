import time
import keras.metrics
import tensorflow as tf
import numpy as np
from keras.losses import BinaryCrossentropy
from keras.metrics import MeanIoU, Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from utils.custom_losses import weighted_binary_crossentropy
from model.unet import get_unet
from utils.constants import POTHOLE_WEIGHT, BACKGROUND_WEIGHT, BATCH_SIZE
from utils.read_data import read_images
from utils.augmentation import generator, generator_with_augmentation
from utils.plot import plot_from_generator, plot_learning_curve
from model.attention_modules import AM

print("Reading train images...")
x_train, y_train = read_images(rgb_path="data/training/rgb/", label_path="data/training/label/")
print("Reading validation images...")
x_valid, y_valid = read_images(rgb_path="data/validation/rgb/", label_path="data/validation/label/")

train_generator = generator_with_augmentation(x=x_train, y=y_train)
valid_generator = generator(x=x_valid, y=y_valid)

# plot_from_generator(generator=train_generator)

model = get_unet(am_scheme=(AM.DUAL, AM.CHANNEL, AM.CHANNEL, AM.CHANNEL, AM.POSITION),
                 n_filters=16,
                 dropout=0.05,
                 batchnorm=True)

# model.compile(optimizer=Adam(learning_rate=1e-4),
#               loss=weighted_binary_crossentropy(POTHOLE_WEIGHT, BACKGROUND_WEIGHT),
#               metrics=[MeanIoU(num_classes=2), "accuracy"])
#
# callbacks = [
#     EarlyStopping(patience=10, verbose=1),
#     ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-8, verbose=1),
#     ModelCheckpoint("model.h5", verbose=1, save_best_only=True),
# ]
#
# results = model.fit(train_generator, epochs=100, callbacks=callbacks, validation_data=valid_generator)
#
# plot_learning_curve(results=results)

# Instantiate one optimizer for the discriminator and another for the generator.
optimizer = Adam(learning_rate=1e-4)

# Instantiate a loss function.
loss_fn = weighted_binary_crossentropy(POTHOLE_WEIGHT, BACKGROUND_WEIGHT)

# Prepare the metrics.
train_acc_metric = MeanIoU(num_classes=2)
val_acc_metric = MeanIoU(num_classes=2)


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        y_pred = tf.where(y_pred >= 0.5, 1, 0)
        loss_value = loss_fn(y, y_pred)
        # Add any extra losses created during the forward pass.
        loss_value += sum(model.losses)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_acc_metric.update_state(y, y_pred)
    return loss_value


@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    y_pred = tf.where(y_pred >= 0.5, 1, 0)
    val_acc_metric.update_state(y, y_pred)


start_time = time.time()
epochs = 50
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    epoch_start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_generator):
        loss_value = train_step(x_batch_train, y_batch_train)

        if step == 1:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training meanIoU over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in valid_generator:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation meanIoU: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - epoch_start_time))

total_time = time.time() - start_time
print()
print("Total time taken: %.2d:%.2d" % (total_time / 60, total_time % 60))

model.save("custom_trained_model.h5")
print("Model saved to \"custom_trained_model.h5\"")
