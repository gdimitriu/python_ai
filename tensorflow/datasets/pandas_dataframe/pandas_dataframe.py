# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2


def get_basic_model():
    model = keras.Sequential([
        normalizer,
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


class MyModel(keras.Model):
    def __init__(self):
        # Create all the internal layers in init.
        super().__init__()

        self.normalizer = keras.layers.Normalization(axis=-1)

        self.seq = keras.Sequential([
            self.normalizer,
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1)
        ])

        self.concat = keras.layers.Concatenate(axis=1)

    def _stack(self, input_dict):
        values = []
        for key, value in sorted(input_dict.items()):
            values.append(value)

        return self.concat(values)

    def adapt(self, inputs):
        # Stack the inputs and `adapt` the normalization layer.
        inputs = self._stack(inputs)
        self.normalizer.adapt(inputs)

    def call(self, inputs):
        # Stack the inputs
        inputs = self._stack(inputs)
        # Run them through all the layers.
        result = self.seq(inputs)

        return result


if __name__ == "__main__":
    csv_file = tf.keras.utils.get_file('heart.csv',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
    df = pd.read_csv(csv_file)
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    print(df.head())
    print(df.dtypes)
    target = df.pop('target')
    # A dataframe as an array
    numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
    numeric_features = df[numeric_feature_names]
    print(numeric_features.head())
    print(tf.convert_to_tensor(numeric_features))
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(numeric_features))
    print(normalizer(numeric_features.iloc[:3]))

    model = get_basic_model()
    model.fit(numeric_features, target, epochs=15, batch_size=BATCH_SIZE)

    # With tf.data
    numeric_dataset = tf.data.Dataset.from_tensor_slices((numeric_features, target))

    for row in numeric_dataset.take(3):
        print(row)

    numeric_batches = numeric_dataset.shuffle(1000).batch(BATCH_SIZE)

    model = get_basic_model()
    model.fit(numeric_batches, epochs=15)

    # A DataFrame as a dictionary
    numeric_features_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in dict(numeric_features).items()}
    target_array = target.to_numpy()[:, tf.newaxis]
    numeric_dict_ds = tf.data.Dataset.from_tensor_slices((numeric_features_dict, target_array))
    print(len(numeric_features_dict))
    for row in numeric_dict_ds.take(3):
        print(row)

    # The Model-subclass style
    model = MyModel()

    model.adapt(numeric_features_dict)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  run_eagerly=True)

    model.fit(numeric_features_dict, target_array, epochs=5, batch_size=BATCH_SIZE)

    numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    model.fit(numeric_dict_batches, epochs=5)
    print(model.predict(dict(numeric_features.iloc[:3])))

    # The Keras functional style
    inputs = {}
    for name, column in numeric_features.items():
        inputs[name] = tf.keras.Input(
            shape=(1,), name=name, dtype=tf.float32)
    print(inputs)

    xs = [value for key, value in sorted(inputs.items())]

    concat = keras.layers.Concatenate(axis=1)
    x = concat(xs)

    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.concatenate([value for key, value in sorted(numeric_features_dict.items())], axis=1))

    x = normalizer(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    model = keras.Model(inputs, x)

    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'],
                  run_eagerly=True)
    keras.utils.plot_model(model, to_file="last_model.png", rankdir="LR", show_shapes=True, show_layer_names=True)
    model.fit(numeric_features_dict, target, epochs=5, batch_size=BATCH_SIZE)
    numeric_dict_batches = numeric_dict_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE)
    model.fit(numeric_dict_batches, epochs=5)
