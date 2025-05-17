# https://www.tensorflow.org/tutorials/load_data/text#example_2_predict_the_author_of_iliad_translations
import collections
import pathlib
from keras import utils
import tensorflow as tf
import tensorflow_text as tf_text
from keras import layers
from keras import losses


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


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

    model = tf.keras.Sequential(my_layers)
    return model


class MyVocabTable(tf.keras.layers.Layer):
    def __init__(self, vocab):
        super().__init__()
        self.keys = [''] + vocab
        self.values = range(len(self.keys))

        self.init = tf.lookup.KeyValueTensorInitializer(
            self.keys, self.values, key_dtype=tf.string, value_dtype=tf.int64)

        num_oov_buckets = 1

        self.table = tf.lookup.StaticVocabularyTable(self.init, num_oov_buckets)

    def call(self, x):
        result = self.table.lookup(x)
        return result


class MyTokenizer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.tokenizer = tf_text.UnicodeScriptTokenizer()

    def call(self, text):
        lower_case = tf_text.case_fold_utf8(text)
        result = self.tokenizer.tokenize(lower_case)
        # If you pass a batch of strings, it will return a RaggedTensor.
        if isinstance(result, tf.RaggedTensor):
            # Convert to dense 0-padded.
            result = result.to_tensor()
        return result


if __name__ == "__main__":
    DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']
    VOCAB_SIZE = 10000

    for name in FILE_NAMES:
        text_dir = utils.get_file(name, origin=DIRECTORY_URL + name)

    parent_dir = pathlib.Path(text_dir).parent
    list(parent_dir.iterdir())
    labeled_data_sets = []

    for i, file_name in enumerate(FILE_NAMES):
        lines_dataset = tf.data.TextLineDataset(str(parent_dir / file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)
    BUFFER_SIZE = 50000
    BATCH_SIZE = 64
    VALIDATION_SIZE = 5000
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

    all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)
    for text, label in all_labeled_data.take(10):
        print("Sentence: ", text.numpy())
        print("Label:", label.numpy())

    tokenizer = MyTokenizer()
    tokenized_ds = all_labeled_data.map(lambda text, label: (tokenizer(text), label))
    print(tokenized_ds)
    for tokens, label in tokenized_ds.take(1):
        break

    print("Tokens:", tokens)
    print()
    print("Label:", label)

    tokenized_ds = tokenized_ds.cache().prefetch(tf.data.AUTOTUNE)

    vocab_count = collections.Counter()
    for toks, labels in tokenized_ds.ragged_batch(1000):
        toks = tf.reshape(toks, [-1])
        for tok in toks.numpy():
            vocab_count[tok] += 1

    vocab = [tok for tok, count in vocab_count.most_common(VOCAB_SIZE)]

    print("First five vocab entries:", vocab[:5])
    print()

    vocab_table = MyVocabTable(['a', 'b', 'c'])
    print(vocab_table(tf.constant([''] + list('abcdefghi'))))

    # vocab_table = MyVocabTable(vocab)
    preprocess_text = tf.keras.Sequential([
        tokenizer,
        vocab_table
    ])
    example_text, example_label = next(iter(all_labeled_data))
    print("Sentence: ", example_text.numpy())
    vectorized_text = preprocess_text(example_text)
    print("Vectorized sentence: ", vectorized_text.numpy())

    all_encoded_data = all_labeled_data.map(lambda text, labels: (preprocess_text(text), labels))

    for ids, label in all_encoded_data.take(1):
        break

    print("Ids: ", ids.numpy())
    print("Label: ", label.numpy())

    train_data = all_encoded_data.skip(VALIDATION_SIZE).shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    validation_data = all_encoded_data.take(VALIDATION_SIZE).padded_batch(BATCH_SIZE)

    sample_text, sample_labels = next(iter(validation_data))
    print("Text batch shape: ", sample_text.shape)
    print("Label batch shape: ", sample_labels.shape)
    print("First text example: ", sample_text[0])
    print("First label example: ", sample_labels[0])

    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

    # Train the model
    model = create_model(vocab_size=VOCAB_SIZE + 2, num_labels=3)

    model.compile(
        optimizer='adam',
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
    history = model.fit(train_data, validation_data=validation_data, epochs=3)

    metrics = model.evaluate(validation_data, return_dict=True)

    print("Loss: ", metrics['loss'])
    print("Accuracy: {:2.2%}".format(metrics['accuracy']))

    # export the model
    export_model = tf.keras.Sequential([
        preprocess_text,
        model
    ])
    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy'])
    # Create a test dataset of raw strings.
    test_ds = all_labeled_data.take(VALIDATION_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    print(test_ds)

    loss, accuracy = export_model.evaluate(test_ds)

    print("Loss: ", loss)
    print("Accuracy: {:2.2%}".format(accuracy))

    tf.saved_model.save(export_model, 'export.tf')
    loaded = tf.saved_model.load('export.tf')
    export_model(tf.constant(['The field bristled with the long and deadly spears which they bore.'])).numpy()
    loaded(tf.constant(['The field bristled with the long and deadly spears which they bore.'])).numpy()

    inputs = [
        "Join'd to th' Ionians with their flowing robes,",  # Label: 1
        "the allies, and his armour flashed about him so that he seemed to all",  # Label: 2
        "And with loud clangor of his arms he fell.",  # Label: 0
    ]

    predicted_scores = export_model.predict(inputs)
    predicted_labels = tf.math.argmax(predicted_scores, axis=1)

    for input, label in zip(inputs, predicted_labels):
        print("Question: ", input)
        print("Predicted label: ", label.numpy())



