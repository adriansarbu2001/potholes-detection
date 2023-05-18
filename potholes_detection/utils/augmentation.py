import albumentations as A
import tensorflow as tf

from potholes_detection.utils.constants import IM_HEIGHT, IM_WIDTH, BATCH_SIZE


transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomResizedCrop(height=IM_HEIGHT, width=IM_WIDTH, scale=(0.5, 1), ratio=(1, 1), p=0.5),
#    A.RandomBrightnessContrast(p=0.5),
])


def zip_generator(x_generator, y_generator):
    xy_generator = tf.data.Dataset \
        .zip((x_generator, y_generator)) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)

    return xy_generator


def zip_generator_with_augmentation(x_generator, y_generator):
    def _augment_images(image, mask):
        transformed = transforms(image=image, mask=mask)

        transformed_image = tf.cast(transformed["image"], tf.float32)
        transformed_mask = tf.cast(transformed["mask"], tf.uint8)

        # tf.print(type(transformed_image), type(transformed_mask))
        return transformed_image, transformed_mask

    def _set_shapes(img, label):
        img.set_shape((IM_HEIGHT, IM_WIDTH, 3))
        label.set_shape((IM_HEIGHT, IM_WIDTH, 1))
        return img, label

    xy_generator = tf.data.Dataset \
        .zip((x_generator, y_generator)) \
        .map(lambda x, y: tf.numpy_function(func=_augment_images, inp=[x, y], Tout=[tf.float32, tf.uint8])) \
        .map(_set_shapes) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.AUTOTUNE)

    return xy_generator
