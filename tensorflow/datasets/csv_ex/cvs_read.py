# https://www.tensorflow.org/tutorials/load_data/csv
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
import keras

if __name__ == "__main__":
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)

    # "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    abalone_train = pd.read_csv(
        "abalone_train.csv",
        names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
               "Viscera weight", "Shell weight", "Age"])

    print(abalone_train.head())
    abalone_features = abalone_train.copy()
    abalone_labels = abalone_features.pop('Age')
    abalone_features = np.array(abalone_features)
    print(abalone_features)

    abalone_model = keras.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                          optimizer=tf.keras.optimizers.Adam())
    history = abalone_model.fit(abalone_features, abalone_labels, epochs=10)
    print(history.history)

    # basic preprocessing
    normalize = layers.Normalization()
    normalize.adapt(abalone_features)
    norm_abalone_model = tf.keras.Sequential([
        normalize,
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    norm_abalone_model.compile(loss=tf.keras.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam())

    norm_history = norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)
    print(norm_history.history)

