# https://www.tensorflow.org/tutorials/images/data_augmentation
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

print(keras.__version__)
print(tf.__version__)
print(tfds.__version__)


def visualize(original, augmented):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Original image')
    plt.imshow(original)

    plt.subplot(1, 2, 2)
    plt.title('Augmented image')
    plt.imshow(augmented)
    plt.show()


if __name__ == "__main__":
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    get_label_name = metadata.features['label'].int2str
    image, label = next(iter(train_ds))
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()
    flipped = tf.image.flip_left_right(image)
    visualize(image, flipped)

    grayscaled = tf.image.rgb_to_grayscale(image)
    visualize(image, tf.squeeze(grayscaled))
    plt.show()

    saturated = tf.image.adjust_saturation(image, 3)
    visualize(image, saturated)

    bright = tf.image.adjust_brightness(image, 0.4)
    visualize(image, bright)

    cropped = tf.image.central_crop(image, central_fraction=0.5)
    visualize(image, cropped)

    rotated = tf.image.rot90(image)
    visualize(image, rotated)

    # Randomly change image brightness
    for i in range(3):
        seed = (i, 0)  # tuple of size (2,)
        stateless_random_brightness = tf.image.stateless_random_brightness(
            image, max_delta=0.95, seed=seed)
        visualize(image, stateless_random_brightness)

    # Randomly change image contrast
    for i in range(3):
        seed = (i, 0)  # tuple of size (2,)
        stateless_random_contrast = tf.image.stateless_random_contrast(
            image, lower=0.1, upper=0.9, seed=seed)
        visualize(image, stateless_random_contrast)

    # Randomly crop an image
    for i in range(3):
        seed = (i, 0)  # tuple of size (2,)
        stateless_random_crop = tf.image.stateless_random_crop(
            image, size=[210, 300, 3], seed=seed)
        visualize(image, stateless_random_crop)

