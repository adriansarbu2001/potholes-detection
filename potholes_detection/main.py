import keras.applications.vgg16
import os
import random
import shutil
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
from utils import plotImages, plot_confusion_matrix


os.chdir("data/roads")
if not os.path.isdir("train"):
    os.makedirs("train/normal")
    os.makedirs("train/potholes")
    os.makedirs("valid/normal")
    os.makedirs("valid/potholes")
    os.makedirs("test/normal")
    os.makedirs("test/potholes")

    normal_count = len(os.listdir("normal"))
    potholes_count = len(os.listdir("potholes"))

    train = 0.7
    valid = 0.2

    for c in random.sample(os.listdir("normal"), int(train * normal_count)):
        shutil.move(f"normal/{c}", "train/normal")
    for c in random.sample(os.listdir("potholes"), int(train * potholes_count)):
        shutil.move(f"potholes/{c}", "train/potholes")
    for c in random.sample(os.listdir("normal"), int(valid * normal_count)):
        shutil.move(f"normal/{c}", "valid/normal")
    for c in random.sample(os.listdir("potholes"), int(valid * potholes_count)):
        shutil.move(f"potholes/{c}", "valid/potholes")
    for c in random.sample(os.listdir("normal"), len(os.listdir("normal"))):
        shutil.move(f"normal/{c}", "test/normal")
    for c in random.sample(os.listdir("potholes"), len(os.listdir("potholes"))):
        shutil.move(f"potholes/{c}", "test/potholes")
os.chdir("../..")

train_path = '../potholes_detection/data/roads/train'
valid_path = '../potholes_detection/data/roads/valid'
test_path = '../potholes_detection/data/roads/test'
batch_size = 10


# train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
train_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=['normal', 'potholes'], batch_size=batch_size)
valid_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=['normal', 'potholes'], batch_size=batch_size)
test_batches = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=['normal', 'potholes'], batch_size=batch_size, shuffle=False)


# imgs, labels = next(train_batches)
# plotImages(imgs, batch_size)
# print(labels)

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax'),
])

# model.summary()
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=2, verbose=2)

# test_imgs, test_labels = next(test_batches)
# plotImages(test_imgs, batch_size)
# print(test_labels)

predictions = model.predict(x=test_batches, verbose=0)
np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))


cm_plot_labels = ['normal', 'potholes']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')
