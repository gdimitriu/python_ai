# https://www.tensorflow.org/tutorials/load_data/text
import collections
import pathlib
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers
from keras import losses
from keras import utils
from keras.layers import TextVectorization
import os

import tensorflow_datasets as tfds
import tensorflow_text as tf_text


# data_url = 'https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'

# dataset_dir = utils.get_file(
#    origin=data_url,
#    untar=True,
#    cache_dir='stack_overflow',
#    cache_subdir='')

# dataset_dir = pathlib.Path(dataset_dir).parent

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Predict tags')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


def create_model(vocab_size, num_labels, vectorizer=None):
    my_layers = []
    if vectorizer is not None:
        my_layers = [vectorizer]

    my_layers.extend([
        layers.Embedding(vocab_size, 64, mask_zero=True),
        layers.Dropout(0.5),
        layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(num_labels)
    ])

    model = keras.Sequential(my_layers)
    return model


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    dataset_dir = pathlib.Path(args.input_dir + "/stack_overflow")
    print(list(dataset_dir.iterdir()))
    train_dir = dataset_dir / 'train'
    list(train_dir.iterdir())
    sample_file = train_dir / 'python/1755.txt'
    with open(sample_file) as f:
        print(f.read())
    batch_size = 32
    seed = 42

    raw_train_ds = utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)
    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(10):
            print("Question: ", text_batch.numpy()[i])
            print("Label:", label_batch.numpy()[i])

    for i, label in enumerate(raw_train_ds.class_names):
        print("Label", i, "corresponds to", label)

    # Create a validation set.
    raw_val_ds = utils.text_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    test_dir = dataset_dir / 'test'

    # Create a test set.
    raw_test_ds = utils.text_dataset_from_directory(
        test_dir,
        batch_size=batch_size)

    raw_train_ds = raw_train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    raw_val_ds = raw_val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    raw_test_ds = raw_test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    VOCAB_SIZE = 10000

    binary_vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='binary')

    MAX_SEQUENCE_LENGTH = 250

    int_vectorize_layer = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH)

    # Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
    train_text = raw_train_ds.map(lambda text, labels: text)
    binary_vectorize_layer.adapt(train_text)
    int_vectorize_layer.adapt(train_text)

    # Retrieve a batch (of 32 reviews and labels) from the dataset.
    text_batch, label_batch = next(iter(raw_train_ds))
    first_question, first_label = text_batch[0], label_batch[0]
    print("Question:", first_question)
    print("Label:", first_label)

    print("'binary' vectorized question:", list(binary_vectorize_layer(first_question).numpy()))

    plt.plot(binary_vectorize_layer(first_question).numpy())
    plt.xlim(0, 1000)
    plt.show()

    print("'int' vectorized question:", int_vectorize_layer(first_question).numpy())

    print("1289 ---> ", int_vectorize_layer.get_vocabulary()[1289])
    print("313 ---> ", int_vectorize_layer.get_vocabulary()[313])
    print("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))

    binary_model = keras.Sequential([
        binary_vectorize_layer,
        layers.Dense(4)])

    binary_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    binary_model.build()
    keras.utils.plot_model(binary_model, to_file="bin_model.png", show_shapes=True)
    bin_history = binary_model.fit(
        raw_train_ds, validation_data=raw_val_ds, epochs=10)

    # `vocab_size` is `VOCAB_SIZE + 1` since `0` is used additionally for padding.
    int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=4, vectorizer=int_vectorize_layer)
    int_model.build()
    keras.utils.plot_model(int_model, to_file="int_model.png", show_shapes=True)
    int_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    int_history = int_model.fit(raw_train_ds, validation_data=raw_val_ds, epochs=10)

    loss = plt.plot(bin_history.epoch, bin_history.history['loss'], label='bin-loss')
    plt.plot(bin_history.epoch, bin_history.history['val_loss'], '--', color=loss[0].get_color(), label='bin-val_loss')

    loss = plt.plot(int_history.epoch, int_history.history['loss'], label='int-loss')
    plt.plot(int_history.epoch, int_history.history['val_loss'], '--', color=loss[0].get_color(), label='int-val_loss')

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('CE/token')
    plt.show()

    binary_train_ds = raw_train_ds.map(lambda x, y: (binary_vectorize_layer(x), y))
    binary_val_ds = raw_val_ds.map(lambda x, y: (binary_vectorize_layer(x), y))
    binary_test_ds = raw_test_ds.map(lambda x, y: (binary_vectorize_layer(x), y))

    int_train_ds = raw_train_ds.map(lambda x, y: (int_vectorize_layer(x), y))
    int_val_ds = raw_val_ds.map(lambda x, y: (int_vectorize_layer(x), y))
    int_test_ds = raw_test_ds.map(lambda x, y: (int_vectorize_layer(x), y))

    binary_model.export('bin.tf')
    loaded = tf.saved_model.load('bin.tf')
    print(binary_model.predict(['How do you sort a list?']))
    print(loaded.serve(tf.constant(['How do you sort a list?'])).numpy())
