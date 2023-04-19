from keras.metrics import MeanIoU, Accuracy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from utils.custom_losses import weighted_binary_crossentropy
from model.unet import get_unet
from utils.constants import POTHOLE_WEIGHT, BACKGROUND_WEIGHT
from utils.read_data import read_images
from utils.augmentation import generator, generator_with_augmentation
from utils.plot import plot_from_generator, plot_learning_curve
from model.attention_modules import AM

print("Reading train images...")
X_train, y_train = read_images(rgb_path="data/training/rgb/", label_path="data/training/label/")
print("Reading validation images...")
X_valid, y_valid = read_images(rgb_path="data/validation/rgb/", label_path="data/validation/label/")

train_generator = generator_with_augmentation(x=X_train, y=y_train)
valid_generator = generator(x=X_valid, y=y_valid)

# plot_from_generator(generator=train_generator)

model = get_unet(am_scheme=(AM.DUAL, AM.CHANNEL, AM.CHANNEL, AM.CHANNEL, AM.POSITION),
                 n_filters=16,
                 dropout=0.05,
                 batchnorm=True)

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=weighted_binary_crossentropy(POTHOLE_WEIGHT, BACKGROUND_WEIGHT),
              metrics=[MeanIoU(num_classes=2), "accuracy"])

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-8, verbose=1),
    ModelCheckpoint("model.h5", verbose=1, save_best_only=True),
]

results = model.fit(train_generator, epochs=100, callbacks=callbacks, validation_data=valid_generator)

plot_learning_curve(results=results)
