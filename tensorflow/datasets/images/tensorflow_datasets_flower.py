# https://www.tensorflow.org/tutorials/load_data/images
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

print(keras.__version__)
print(tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


if __name__ == "__main__":
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'tf_flowers',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        shuffle_files=True,
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

    train_ds = configure_for_performance(train_ds)
    val_ds = configure_for_performance(val_ds)
    test_ds = configure_for_performance(test_ds)
