# https://www.tensorflow.org/tutorials/images/data_augmentation
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


def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


def random_invert_img(x, p=0.5):
    if tf.random.uniform([]) < p:
        x = (255 - x)
    else:
        x
    return x


def random_invert(factor=0.5):
    return layers.Lambda(lambda x: random_invert_img(x, factor))


class RandomInvert(layers.Layer):
    def __init__(self, factor=0.5, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return random_invert_img(x)


if __name__ == "__main__":
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    num_classes = metadata.features['label'].num_classes
    print(num_classes)
    get_label_name = metadata.features['label'].int2str

    image, label = next(iter(train_ds))
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()

    IMG_SIZE = 180

    resize_and_rescale = keras.Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1. / 255)
    ])

    result = resize_and_rescale(image)
    plt.imshow(result)
    plt.show()

    print("Min and max pixel values:", result.numpy().min(), result.numpy().max())
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ])

    # Add the image to a batch.
    image = tf.cast(tf.expand_dims(image, 0), tf.float32)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0])
        plt.axis("off")
    plt.show()

    # option 1
    # model = keras.Sequential([
    #        Add the preprocessing layers you created earlier.
    #        resize_and_rescale,
    #        data_augmentation,
    #        layers.Conv2D(16, 3, padding='same', activation='relu'),
    #        layers.MaxPooling2D(),
    #        Rest of your model.
    #    ])
    # option 2
    # aug_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))

    train_ds = prepare(train_ds, shuffle=True, augment=True)
    val_ds = prepare(val_ds)
    test_ds = prepare(test_ds)

    # train the model
    model = keras.Sequential([
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    epochs = 5
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    loss, acc = model.evaluate(test_ds)
    print("Accuracy", acc)

    random_invert = random_invert()

    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_image = random_invert(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()
    plt.imshow(RandomInvert()(image)[0])
    plt.show()

