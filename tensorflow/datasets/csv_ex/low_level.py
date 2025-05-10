# https://www.tensorflow.org/tutorials/load_data/csv
import pathlib
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

if __name__ == "__main__":
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    text = pathlib.Path("titanic.csv").read_text()
    lines = text.split('\n')[1:-1]

    all_strings = [str()] * 10
    print(all_strings)

    features = tf.io.decode_csv(lines, record_defaults=all_strings)
    for f in features:
        print(f"type: {f.dtype.name}, shape: {f.shape}")
    print(lines[0])
    titanic_types = [int(), str(), float(), int(), int(), float(), str(), str(), str(), str()]
    print(titanic_types)
    features = tf.io.decode_csv(lines, record_defaults=titanic_types)
    for f in features:
        print(f"type: {f.dtype.name}, shape: {f.shape}")


    print("Reading again")
    simple_titanic = tf.data.experimental.CsvDataset("titanic.csv", record_defaults=titanic_types, header=True)
    for example in simple_titanic.take(1):
        print([e.numpy() for e in example])

