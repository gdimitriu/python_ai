# https://www.tensorflow.org/tutorials/load_data/csv
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
import keras
import os
import itertools


def slices(features):
  for i in itertools.count():
    # For each feature take index `i`
    example = {name:values[i] for name, values in features.items()}
    yield example


def titanic_model(preprocessing_head, inputs):
  body = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = keras.Model(inputs, result)

  model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=keras.optimizers.Adam())
  return model


if __name__ == "__main__":
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # MIXED Data types
    #titanic_file_path = keras.utils.get_file("titanic.csv",
    #                "https://storage.googleapis.com/tf-datasets/titanic/train.csv")
    # https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    titanic = pd.read_csv("titanic.csv")
    print(titanic.head())
    titanic_features = titanic.copy()
    titanic_labels = titanic_features.pop('survived')
    # Create a symbolic input
    input = keras.Input(shape=(), dtype=tf.float32)

    # Perform a calculation using the input
    result = 2 * input + 1

    # the result doesn't have a value
    print(result)
    calc = keras.Model(inputs=input, outputs=result)
    print(calc(np.array([1])).numpy())
    print(calc(np.array([2])).numpy())

    inputs = {}

    for name, column in titanic_features.items():
        dtype = column.dtype
        if dtype == object:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = keras.Input(shape=(1,), name=name, dtype=dtype)

    print(inputs)

    numeric_inputs = {name: input for name, input in inputs.items()
                      if input.dtype == tf.float32}

    x = layers.Concatenate()(list(numeric_inputs.values()))
    norm = layers.Normalization()
    norm.adapt(np.array(titanic[numeric_inputs.keys()]))
    all_numeric_inputs = norm(x)

    print(all_numeric_inputs)
    # Collect all the symbolic preprocessing results, to concatenate them later:
    preprocessed_inputs = [all_numeric_inputs]

    for name, input in inputs.items():
        if input.dtype == tf.float32:
            continue

        lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
        one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

        x = lookup(input)
        x = one_hot(x)
        preprocessed_inputs.append(x)

    preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

    titanic_preprocessing = keras.Model(inputs, preprocessed_inputs_cat)

    keras.utils.plot_model(model=titanic_preprocessing, rankdir="LR", dpi=72, show_shapes=True)

    titanic_features_dict = {name: np.array(value) for name, value in titanic_features.items()}
    features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}
    print(titanic_preprocessing(features_dict))
    # Build the model
    titanic_model = titanic_model(titanic_preprocessing, inputs)
    history = titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)
    print(history.history)

    titanic_model.save('test.keras')
    reloaded = keras.models.load_model('test.keras')

    features_dict = {name: values[:1] for name, values in titanic_features_dict.items()}

    before = titanic_model(features_dict)
    after = reloaded(features_dict)
    assert (before - after) < 1e-3
    print(before)
    print(after)
    os.remove("test.keras")

    for example in slices(titanic_features_dict):
        for name, value in example.items():
            print(f"{name:19s}: {value}")
        break

    print("now a tensor")
    features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)
    for example in features_ds:
        for name, value in example.items():
            print(f"{name:19s}: {value}")
        break

    titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
    titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)
    titanic_model.fit(titanic_batches, epochs=5)

    print("CSV dataset")
    titanic_csv_ds = tf.data.experimental.make_csv_dataset(
        "titanic.csv",
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name='survived',
        num_epochs=1,
        ignore_errors=True, )
    for batch, label in titanic_csv_ds.take(1):
        for key, value in batch.items():
            print(f"{key:20s}: {value}")
        print()
        print(f"{'label':20s}: {label}")
    