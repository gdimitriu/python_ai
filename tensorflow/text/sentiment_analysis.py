import tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers
from keras import losses


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


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


if __name__ == "__main__":
    BATCH_SIZE = 64
    VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 1000

    # Training set.
    train_ds = tfds.load(
        'imdb_reviews',
        split='train[:80%]',
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        as_supervised=True)
    # Validation set.
    val_ds = tfds.load(
        'imdb_reviews',
        split='train[80%:]',
        batch_size=BATCH_SIZE,
        shuffle_files=True,
        as_supervised=True)

    for review_batch, label_batch in val_ds.take(1):
        for i in range(5):
            print("Review: ", review_batch[i].numpy())
            print("Label: ", label_batch[i].numpy())

    vectorize_layer = layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH)

    # Make a text-only dataset (without labels), then call `TextVectorization.adapt`.
    train_text = train_ds.map(lambda text, labels: text)
    vectorize_layer.adapt(train_text)
    train_ds = train_ds.map(vectorize_text)
    val_ds = val_ds.map(vectorize_text)
    # Configure datasets for performance as before.
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    model = create_model(vocab_size=VOCAB_SIZE, num_labels=1)
    print(model.summary())
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer='adam',
        metrics=['accuracy'])
    history = model.fit(train_ds, validation_data=val_ds, epochs=3)
    loss, accuracy = model.evaluate(val_ds)

    print("Loss: ", loss)
    print("Accuracy: {:2.2%}".format(accuracy))

    export_model = tf.keras.Sequential(
        [vectorize_layer, model,
         layers.Activation('sigmoid')])

    export_model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer='adam',
        metrics=['accuracy'])
    # 0 --> negative review
    # 1 --> positive review
    inputs = [
        "This is a fantastic movie.",
        "This is a bad movie.",
        "This movie was so bad that it was good.",
        "I will never say yes to watching this movie.",
    ]
    #batch_utf8 = [s.encode('UTF-8') for s in inputs]
    #batch_chars_ragged = tf.strings.unicode_decode(batch_utf8[0], input_encoding='UTF-8')
    #batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
    # does not work
    predicted_scores = export_model.predict(inputs)
    predicted_labels = [int(round(x[0])) for x in predicted_scores]

    for input, label in zip(inputs, predicted_labels):
        print("Question: ", input)
        print("Predicted label: ", label)
