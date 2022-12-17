import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize

from keras.models import load_model
from keras.utils import img_to_array, load_img

plt.style.use("ggplot")

# Set some parameters
im_width = 400
im_height = 400
border = 5

model = load_model('model.h5')

# model.summary()

# load the best model
model.load_weights('model.h5')

img = load_img("test.png", color_mode='rgb')
label = load_img("test_real_label.png", color_mode='grayscale')
x_img = img_to_array(img)
x_img = resize(x_img, (400, 400, 3), mode='constant', preserve_range=True)
x_img = x_img / 255.0
label = img_to_array(label)
label = resize(label, (400, 400, 1), mode='constant', preserve_range=True)
label = label / 255.0

res = model.predict(np.array([x_img]))
res = np.where(res > 0.5, 1, res)
res = np.where(res < 0.5, 0, res)

has_mask = x_img.max() > 0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 15))

ax1.imshow(x_img)
if has_mask:
    # draw a boundary(contour) in the original image separating pothole and background areas
    ax1.contour(res[0].squeeze(), colors='k', linewidths=5, levels=[0.5])
ax1.set_title('Test image')

ax2.imshow(res[0].squeeze(), cmap='gray', interpolation='bilinear')
ax2.set_title('Predicted')
# ax3.imshow(label.squeeze(), cmap='gray', interpolation='bilinear')
# ax3.set_title('Real label')
plt.show()
