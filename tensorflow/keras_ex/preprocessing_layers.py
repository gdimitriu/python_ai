# https://www.tensorflow.org/guide/keras/preprocessing_layers
import numpy as np
import tensorflow as tf
import keras
from keras import layers


def Quick_recipes():
    import keras
    from keras import layers
    # image data augumentation
    # Create a data augmentation stage with horizontal flipping, rotations, zooms
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    # Load some data
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    input_shape = x_train.shape[1:]
    classes = 10

    # Create a tf.data pipeline of augmented images (and their labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))

    # Create a model and train it on the augmented image data
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)  # Rescale inputs
    outputs = keras.applications.ResNet50(  # Add the rest of the model
        weights=None, input_shape=input_shape, classes=classes
    )(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    model.fit(train_dataset, steps_per_epoch=5)

    # normalizing numerical features
    # Load some data
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.reshape((len(x_train), -1))
    input_shape = x_train.shape[1:]
    classes = 10

    # Create a Normalization layer and set its internal state using the training data
    normalizer = layers.Normalization()
    normalizer.adapt(x_train)

    # Create a model that include the normalization layer
    inputs = keras.Input(shape=input_shape)
    x = normalizer(inputs)
    outputs = layers.Dense(classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # Train the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(x_train, y_train)

    # Encoding string categorical features via one-hot encoding
    # Define some toy data
    data = tf.constant([["a"], ["b"], ["c"], ["b"], ["c"], ["a"]])

    # Use StringLookup to build an index of the feature values and encode output.
    lookup = layers.StringLookup(output_mode="one_hot")
    lookup.adapt(data)

    # Convert new test data (which includes unknown feature values)
    test_data = tf.constant([["a"], ["b"], ["c"], ["d"], ["e"], [""]])
    encoded_data = lookup(test_data)
    print(encoded_data)

    # Encoding integer categorical features via one-hot encoding
    # Define some toy data
    data = tf.constant([[10], [20], [20], [10], [30], [0]])

    # Use IntegerLookup to build an index of the feature values and encode output.
    lookup = layers.IntegerLookup(output_mode="one_hot")
    lookup.adapt(data)

    # Convert new test data (which includes unknown feature values)
    test_data = tf.constant([[10], [10], [20], [50], [60], [0]])
    encoded_data = lookup(test_data)
    print(encoded_data)

    # Applying the hashing trick to an integer categorical feature
    # Sample data: 10,000 random integers with values between 0 and 100,000
    data = np.random.randint(0, 100000, size=(10000, 1))

    # Use the Hashing layer to hash the values to the range [0, 64]
    hasher = layers.Hashing(num_bins=64, salt=1337)

    # Use the CategoryEncoding layer to multi-hot encode the hashed values
    encoder = layers.CategoryEncoding(num_tokens=64, output_mode="multi_hot")
    encoded_data = encoder(hasher(data))
    print(encoded_data.shape)

    # Encoding text as a sequence of token indices
    # Define some text data to adapt the layer
    adapt_data = tf.constant(
        [
            "The Brain is wider than the Sky",
            "For put them side by side",
            "The one the other will contain",
            "With ease and You beside",
        ]
    )

    # Create a TextVectorization layer
    text_vectorizer = layers.TextVectorization(output_mode="int")
    # Index the vocabulary via `adapt()`
    text_vectorizer.adapt(adapt_data)

    # Try out the layer
    print(
        "Encoded text:\n",
        text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
    )

    # Create a simple model
    inputs = keras.Input(shape=(None,), dtype="int64")
    x = layers.Embedding(input_dim=text_vectorizer.vocabulary_size(), output_dim=16)(inputs)
    x = layers.GRU(8)(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    # Create a labeled dataset (which includes unknown tokens)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
    )

    # Preprocess the string inputs, turning them into int sequences
    train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
    # Train the model on the int sequences
    print("\nTraining model...")
    model.compile(optimizer="rmsprop", loss="mse")
    model.fit(train_dataset)

    # For inference, you can export a model that accepts strings as input
    inputs = keras.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    outputs = model(x)
    end_to_end_model = keras.Model(inputs, outputs)

    # Call the end-to-end model on test data (which includes unknown tokens)
    print("\nCalling end-to-end model on test string...")
    test_data = tf.constant(["The one the other will absorb"])
    test_output = end_to_end_model(test_data)
    print("Model output:", test_output)

    # Encoding text as a dense matrix of N-grams with multi-hot encoding
    # Define some text data to adapt the layer
    adapt_data = tf.constant(
        [
            "The Brain is wider than the Sky",
            "For put them side by side",
            "The one the other will contain",
            "With ease and You beside",
        ]
    )
    # Instantiate TextVectorization with "multi_hot" output_mode
    # and ngrams=2 (index all bigrams)
    text_vectorizer = layers.TextVectorization(output_mode="multi_hot", ngrams=2)
    # Index the bigrams via `adapt()`
    text_vectorizer.adapt(adapt_data)

    # Try out the layer
    print(
        "Encoded text:\n",
        text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
    )

    # Create a simple model
    inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
    outputs = layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)

    # Create a labeled dataset (which includes unknown tokens)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
    )

    # Preprocess the string inputs, turning them into int sequences
    train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
    # Train the model on the int sequences
    print("\nTraining model...")
    model.compile(optimizer="rmsprop", loss="mse")
    model.fit(train_dataset)

    # For inference, you can export a model that accepts strings as input
    inputs = keras.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    outputs = model(x)
    end_to_end_model = keras.Model(inputs, outputs)

    # Call the end-to-end model on test data (which includes unknown tokens)
    print("\nCalling end-to-end model on test string...")
    test_data = tf.constant(["The one the other will absorb"])
    test_output = end_to_end_model(test_data)
    print("Model output:", test_output)

    # Encoding text as a dense matrix of N-grams with TF-IDF weighting
    # Define some text data to adapt the layer
    adapt_data = tf.constant(
        [
            "The Brain is wider than the Sky",
            "For put them side by side",
            "The one the other will contain",
            "With ease and You beside",
        ]
    )
    # Instantiate TextVectorization with "tf-idf" output_mode
    # (multi-hot with TF-IDF weighting) and ngrams=2 (index all bigrams)
    text_vectorizer = layers.TextVectorization(output_mode="tf-idf", ngrams=2)
    # Index the bigrams and learn the TF-IDF weights via `adapt()`
    text_vectorizer.adapt(adapt_data)

    # Try out the layer
    print(
        "Encoded text:\n",
        text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
    )

    # Create a simple model
    inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
    outputs = layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)

    # Create a labeled dataset (which includes unknown tokens)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
    )

    # Preprocess the string inputs, turning them into int sequences
    train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
    # Train the model on the int sequences
    print("\nTraining model...")
    model.compile(optimizer="rmsprop", loss="mse")
    model.fit(train_dataset)

    # For inference, you can export a model that accepts strings as input
    inputs = keras.Input(shape=(1,), dtype="string")
    x = text_vectorizer(inputs)
    outputs = model(x)
    end_to_end_model = keras.Model(inputs, outputs)

    # Call the end-to-end model on test data (which includes unknown tokens)
    print("\nCalling end-to-end model on test string...")
    test_data = tf.constant(["The one the other will absorb"])
    test_output = end_to_end_model(test_data)
    print("Model output:", test_output)

if __name__ == "__main__":
    data = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.8, 0.9, 1.0],
            [1.5, 1.6, 1.7],
        ]
    )
    layer = layers.Normalization()
    layer.adapt(data)
    normalized_data = layer(data)
    print("Features mean: %.2f" % (normalized_data.numpy().mean()))
    print("Features std: %.2f" % (normalized_data.numpy().std()))

    data = [
        "ξεῖν᾽, ἦ τοι μὲν ὄνειροι ἀμήχανοι ἀκριτόμυθοι",
        "γίγνοντ᾽, οὐδέ τι πάντα τελείεται ἀνθρώποισι.",
        "δοιαὶ γάρ τε πύλαι ἀμενηνῶν εἰσὶν ὀνείρων:",
        "αἱ μὲν γὰρ κεράεσσι τετεύχαται, αἱ δ᾽ ἐλέφαντι:",
        "τῶν οἳ μέν κ᾽ ἔλθωσι διὰ πριστοῦ ἐλέφαντος,",
        "οἵ ῥ᾽ ἐλεφαίρονται, ἔπε᾽ ἀκράαντα φέροντες:",
        "οἱ δὲ διὰ ξεστῶν κεράων ἔλθωσι θύραζε,",
        "οἵ ῥ᾽ ἔτυμα κραίνουσι, βροτῶν ὅτε κέν τις ἴδηται.",
    ]
    layer = layers.TextVectorization()
    layer.adapt(data)
    vectorized_text = layer(data)
    print(vectorized_text)

    vocab = ["a", "b", "c", "d"]
    data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
    layer = layers.StringLookup(vocabulary=vocab)
    vectorized_data = layer(data)
    print(vectorized_data)
    Quick_recipes()

