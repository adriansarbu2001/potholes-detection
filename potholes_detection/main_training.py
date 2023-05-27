import time
import tensorflow as tf
import numpy as np
from keras.metrics import MeanIoU, Accuracy, Mean
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

from model.model import get_unet, get_optimizer, get_loss_fn
from utils.read_data import read_images
from utils.augmentation import zip_generator, zip_generator_with_augmentation
from utils.plot import plot_from_generator, plot_learning_curve
from model.attention_modules import AM
import matplotlib.pyplot as plt

print("Tensorflow version", tf.__version__)
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')), '\n')


print("Reading train images...")
x_train_generator, y_train_generator = read_images(rgb_path="data/pothole600/training/rgb/",
                                                   label_path="data/pothole600/training/label/")
print("Reading validation images...")
x_valid_generator, y_valid_generator = read_images(rgb_path="data/pothole600/validation/rgb/",
                                                   label_path="data/pothole600/validation/label/")

train_generator = zip_generator_with_augmentation(x_generator=x_train_generator, y_generator=y_train_generator)
valid_generator = zip_generator(x_generator=x_valid_generator, y_generator=y_valid_generator)

# plot_from_generator(generator=train_generator)

model = get_unet(am_scheme=(AM.POSITION, AM.CHANNEL, AM.CHANNEL, AM.CHANNEL, AM.DUAL),
                 n_filters=16,
                 dropout=0.05,
                 batchnorm=True)

# model.compile(optimizer=Adam(learning_rate=1e-4),
#               loss=weighted_binary_crossentropy(POTHOLE_WEIGHT, BACKGROUND_WEIGHT),
#               metrics=[MeanIoU(num_classes=2), "accuracy"])
#
# results = model.fit(train_generator, epochs=100, callbacks=callbacks, validation_data=valid_generator)
#
# plot_learning_curve(results=results)

optimizer = get_optimizer()

# Instantiate a loss function.
loss_fn = get_loss_fn()

# Prepare the metrics.
train_loss_metric = Mean(name='train_loss')
val_loss_metric = Mean(name='val_loss')
train_iou_metric = MeanIoU(num_classes=2)
val_iou_metric = MeanIoU(num_classes=2)


@tf.function
def train_step(x, y):
    # Forward pass
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss_value = loss_fn(y, y_pred)
        # Add any extra losses created during the forward pass.
        # loss_value += sum(model.losses)

    # Backward pass
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    y_pred = tf.where(y_pred >= 0.5, 1.0, 0.0)
    train_loss_metric.update_state(loss_value)
    train_iou_metric.update_state(y, y_pred)


@tf.function
def test_step(x, y):
    y_pred = model(x, training=False)
    loss_value = loss_fn(y, y_pred)
    y_pred = tf.where(y_pred >= 0.5, 1.0, 0.0)

    val_loss_metric.update_state(loss_value)
    val_iou_metric.update_state(y, y_pred)


# _callbacks = [
#     EarlyStopping(patience=10, verbose=1),
#     ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-8, verbose=1),
#     ModelCheckpoint("custom_trained_model_2.h5", verbose=1, save_best_only=True),
# ]
#
# callbacks = tf.keras.callbacks.CallbackList(
#     _callbacks, add_history=True, model=model)
#
# logs = {}
# callbacks.on_train_begin(logs=logs)

early_stopping_count = 0
reduce_lr_count = 0
early_stopping_patience = 30
reduce_lr_patience = 10
reduce_lr_factor = 0.5
min_lr = 1e-8
max_epochs = 500
loss_history = []
start_time = time.time()
metrics_history = {"train_loss": [], "train_iou": [], "valid_loss": [], "valid_iou": []}

for epoch in range(max_epochs):
    # callbacks.on_epoch_begin(epoch, logs=logs)
    print("\nStart of epoch %d" % (epoch,))
    print("Current learning rate:", optimizer.learning_rate.numpy())
    epoch_start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_generator):
        # callbacks.on_batch_begin(step, logs=logs)
        # callbacks.on_train_batch_begin(step, logs=logs)

        train_step(x_batch_train, y_batch_train)

        # callbacks.on_train_batch_end(step, logs=logs)
        # callbacks.on_batch_end(step, logs=logs)

    # Display metrics at the end of each epoch.
    train_iou = train_iou_metric.result()
    train_loss = train_loss_metric.result()
    metrics_history["train_loss"].append(train_loss)
    metrics_history["train_iou"].append(train_iou)
    print("Training loss over epoch: %.4f" % (float(train_loss),))
    print("Training meanIoU over epoch: %.4f" % (float(train_iou),))

    train_iou_metric.reset_states()
    train_loss_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in valid_generator:
        test_step(x_batch_val, y_batch_val)

    # callbacks.on_epoch_end(epoch, logs=logs)

    val_loss = val_loss_metric.result()
    val_iou = val_iou_metric.result()
    metrics_history["valid_loss"].append(val_loss)
    metrics_history["valid_iou"].append(val_iou)
    print("Validation loss over epoch: %.4f" % (float(val_loss),))
    print("Validation meanIoU over epoch: %.4f" % (float(val_iou),))

    val_iou_metric.reset_states()
    val_loss_metric.reset_states()

    if len(loss_history) > 0 and val_loss < min(loss_history):
        loss_history = [val_loss]
        early_stopping_count = 1
        reduce_lr_count = 1
    else:
        loss_history.append(val_loss)
        early_stopping_count += 1
        reduce_lr_count += 1

    # print(model.stop_training)
    # if model.stop_training:
    #     break

    # Model checkpoint callback
    if len(loss_history) == 1:
        model.save("custom_trained_model.h5")
        print("Model saved to \"custom_trained_model.h5\"")

    # Reduce learning rate callback
    if optimizer.learning_rate > min_lr and reduce_lr_count > reduce_lr_patience:
        # K.set_value(optimizer.learning_rate, optimizer.learning_rate.numpy() * reduce_lr_factor)

        # load the best model
        # print("Loading best model...")
        # best_model = load_model("custom_trained_model.h5", compile=False)
        # model.set_weights(best_model.weights)

        new_lr = optimizer.learning_rate.numpy() * reduce_lr_factor
        reduce_lr_count = 0
        print(f'Reducing learning rate to {new_lr}. No improvement in '
              f'validation loss in the last {reduce_lr_patience} epochs.')
        optimizer.learning_rate = new_lr

    # Early stopping callback
    if early_stopping_count > early_stopping_patience:
        early_stopping_count = 0
        print(f'\nEarly stopping. No improvement in validation '
              f'loss in the last {early_stopping_patience} epochs.')
        print("Time taken over epoch: %.2fs" % (time.time() - epoch_start_time))
        break
    print("Time taken over epoch: %.2fs" % (time.time() - epoch_start_time))

# callbacks.on_train_end(logs=logs)

print()
total_time = time.time() - start_time
print("Total time taken: %.2d:%.2d:%.2d" % (total_time / 3600, (total_time / 60) % 60, total_time % 60))

plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(metrics_history["train_loss"], label="Train Loss")
plt.plot(metrics_history["valid_loss"], label="Validation Loss")
plt.plot(np.argmin(metrics_history["valid_loss"]),
         metrics_history["valid_loss"][np.argmin(metrics_history["valid_loss"])], marker="x", color="r",
         label="Saved model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8, 8))
plt.title("MeanIoU metric")
plt.plot(metrics_history["train_iou"], label="Train meanIoU")
plt.plot(metrics_history["valid_iou"], label="Validation meanIoU")
plt.plot(np.argmin(metrics_history["valid_loss"]),
         metrics_history["valid_iou"][np.argmin(metrics_history["valid_loss"])], marker="x", color="r",
         label="Saved model")
plt.xlabel("Epochs")
plt.ylabel("meanIoU")
plt.legend()
plt.show()
