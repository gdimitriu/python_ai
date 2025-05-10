# https://www.tensorflow.org/tutorials/load_data/images
import argparse
import pathlib
import PIL.Image
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

print(keras.__version__)
print(tf.__version__)




# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
# data_dir = pathlib.Path(archive).with_suffix('')

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Kmeans \
            using SalesTransaction data')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    data_dir = pathlib.Path(args.input_dir).with_suffix("")
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    roses = list(data_dir.glob('roses/*'))
    image = PIL.Image.open(str(roses[0]))
    image.show()
    roses = list(data_dir.glob('roses/*'))
    image = PIL.Image.open(str(roses[1]))
    image.show()
    print("Height=", image.height)
    print("Width=", image.width)

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    # Standardize the data
    normalization_layer = keras.layers.Rescaling(1. / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # train a model
    num_classes = 5

    model = keras.Sequential([
        keras.layers.Rescaling(1. / 255),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )
