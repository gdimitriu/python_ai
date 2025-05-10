# https://www.tensorflow.org/text/tutorials/warmstart_embedding_matrix
import argparse
import os
import re
import string

import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Translate pt in en with transformers')
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
    # dataset = keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True, cache_dir=".", cache_subdir="")

    dataset_dir = os.path.join(args.input_dir, "aclImdb")
    print(os.listdir(dataset_dir))
    train_dir = os.path.join(dataset_dir, "train")
    print(os.listdir(train_dir))
    # I have already downloaded and removed
    # remove_dir = os.path.join(train_dir, "unsup")
    # shutil.rmtree(remove_dir)
    batch_size = 1024
    seed = 123
    train_ds = keras.utils.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2, subset="training", seed=seed)
    val_ds = keras.utils.text_dataset_from_directory(
        train_dir, batch_size=batch_size, validation_split=0.2, subset="validation", seed=seed)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Make a text-only dataset (no labels) and call `Dataset.adapt` to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # CREATE A CLASSIFICATION MODEL
    embedding_dim = 16
    text_embedding = Embedding(vocab_size, embedding_dim, name="embedding")
    text_input = keras.Sequential(
        [vectorize_layer, text_embedding], name="text_input"
    )
    classifier_head = keras.Sequential(
        [GlobalAveragePooling1D(), Dense(16, activation="relu"), Dense(1)],
        name="classifier_head",
    )

    model = keras.Sequential([text_input, classifier_head])
    # COMPILE AND TRAIN
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="logs")
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[tensorboard_callback],
    )
    model.summary()

    # VOCABULARY REMAPPING
    embedding_weights_base = (
        model.get_layer("text_input").get_layer("embedding").embeddings
    )
    vocab_base = vectorize_layer.get_vocabulary()
    # Vocabulary size and number of words in a sequence.
    vocab_size_new = 10200
    sequence_length = 100

    vectorize_layer_new = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size_new,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer_new.adapt(text_ds)

    # Get the new vocabulary
    vocab_new = vectorize_layer_new.get_vocabulary()
    # View the new vocabulary tokens that weren't in `vocab_base`
    print(set(vocab_base) ^ set(vocab_new))

    # Generate the updated embedding matrix
    updated_embedding = tf.keras.utils.warmstart_embedding_matrix(
        base_vocabulary=vocab_base,
        new_vocabulary=vocab_new,
        base_embeddings=embedding_weights_base,
        new_embeddings_initializer="uniform",
    )
    # Update the model variable
    updated_embedding_variable = tf.Variable(updated_embedding)
    # OR
    # generate updated embedding matrix
    # new_embedding = np.random.rand(len(vocab_new), 16)
    # updated_embedding = tf.keras.utils.warmstart_embedding_matrix(
    #    base_vocabulary=vocab_base,
    #    new_vocabulary=vocab_new,
    #    base_embeddings=embedding_weights_base,
    #    new_embeddings_initializer=tf.keras.initializers.Constant(
    #        new_embedding
    #    )
    # )
    # update model variable
    # updated_embedding_variable = tf.Variable(updated_embedding)

    print(updated_embedding_variable.shape)

    text_embedding_layer_new = Embedding(
        vectorize_layer_new.vocabulary_size(), embedding_dim, name="embedding"
    )
    text_embedding_layer_new.build(input_shape=[None])
    text_embedding_layer_new.embeddings.assign(updated_embedding)
    text_input_new = tf.keras.Sequential(
        [vectorize_layer_new, text_embedding_layer_new], name="text_input_new"
    )
    text_input_new.summary()

    # Verify the shape of updated weights
    # The new weights shape should reflect the new vocabulary size
    print(text_input_new.get_layer("embedding").embeddings.shape)
    warm_started_model = keras.Sequential([text_input_new, classifier_head])
    warm_started_model.summary()

    # New vocab words
    base_vocab_index = vectorize_layer("the")[0]
    new_vocab_index = vectorize_layer_new("the")[0]
    print(
        warm_started_model.get_layer("text_input_new").get_layer("embedding")(
            new_vocab_index
        )
        == embedding_weights_base[base_vocab_index]
    )

    # CONTINUE WITH WARM-STARTED TRAINING
    model.compile(
        optimizer="adam",
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[tensorboard_callback],
    )
