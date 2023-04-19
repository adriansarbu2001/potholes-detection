from keras.models import load_model

from utils.custom_losses import weighted_binary_crossentropy
from utils.read_data import read_images
from utils.augmentation import generator

x, y = read_images(rgb_path="data/testing/rgb/", label_path="data/testing/label/")
test_generator = generator(x=x, y=y)

# load the best model
model = load_model("model.h5", custom_objects={"loss": weighted_binary_crossentropy(0.92, 0.08)})

# model.summary()

# Evaluate on validation set (this must be equals to the best log_loss)
model.evaluate(test_generator, verbose=1)
