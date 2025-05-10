# https://www.tensorflow.org/text/tutorials/word_embeddings
import argparse
import io
import os
import re
import string

import keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


def build_arg_parser():
    parser = argparse.ArgumentParser(description='word embedding imdb')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


# Create a custom standardization function to strip HTML break tags '<br />'.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    # url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    # dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir='.', cache_subdir='')
    dataset = args.input_dir
    dataset_dir = os.path.join(dataset, 'aclImdb')
    print(os.listdir(dataset_dir))
    train_dir = os.path.join(dataset_dir, 'train')
    print(os.listdir(train_dir))
    # remove_dir = os.path.join(train_dir, 'unsup')
    # shutil.rmtree(remove_dir)
    batch_size = 1024
    seed = 123
    train_ds = keras.utils.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2, subset='training', seed=seed)
    val_ds = keras.utils.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2, subset='validation', seed=seed)
    for text_batch, label_batch in train_ds.take(1):
        for i in range(5):
            print(label_batch[i].numpy(), text_batch.numpy()[i])
    # prepare the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # EMBEDDING LAYER
    # Embed a 1,000 word vocabulary into 5 dimensions.
    embedding_layer = keras.layers.Embedding(1000, 5)
    result = embedding_layer(tf.constant([1, 2, 3]))
    print(result.numpy())
    result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
    print(result.shape)

    # TEXT PREPROCESSING
    # Vocabulary size and number of words in a sequence.
    vocab_size = 10000
    sequence_length = 100

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)
    # CREATE A CLASSIFICATION MODEL
    embedding_dim = 16
    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[tensorboard_callback])
    model.summary()

    weights = model.get_layer('embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
