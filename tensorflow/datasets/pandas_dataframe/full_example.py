# https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

SHUFFLE_BUFFER = 500
BATCH_SIZE = 2

if __name__ == "__main__":
    csv_file = tf.keras.utils.get_file('heart.csv',
                                       'https://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
    df = pd.read_csv(csv_file)
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    target = df.pop('target')
    binary_feature_names = ['sex', 'fbs', 'exang']
    categorical_feature_names = ['cp', 'restecg', 'slope', 'thal', 'ca']
    inputs = {}
    for name, column in df.items():
        if type(column[0]) == str:
            dtype = tf.string
        elif (name in categorical_feature_names or
              name in binary_feature_names):
            dtype = tf.int64
        else:
            dtype = tf.float32

        inputs[name] = keras.Input(shape=(1,), name=name, dtype=dtype)
    print(inputs)
    preprocessed = []

    for name in binary_feature_names:
        inp = inputs[name]
        preprocessed.append(inp)
    print(preprocessed)

    numeric_feature_names = ['age', 'thalach', 'trestbps', 'chol', 'oldpeak']
    numeric_features = df[numeric_feature_names]
    numeric_features_dict = {key: value.to_numpy()[:, tf.newaxis] for key, value in dict(numeric_features).items()}
    normalizer = keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.concatenate([value for key, value in sorted(numeric_features_dict.items())], axis=1))
    numeric_inputs = []
    for name in numeric_feature_names:
        numeric_inputs.append(inputs[name])

    numeric_inputs = keras.layers.Concatenate(axis=-1)(numeric_inputs)
    numeric_normalized = normalizer(numeric_inputs)

    preprocessed.append(numeric_normalized)
    print(preprocessed)

    # Categorical features
    vocab = ['a', 'b', 'c']
    lookup = keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
    print(lookup(['c', 'a', 'a', 'b', 'zzz']))
    vocab = [1, 4, 7, 99]
    lookup = keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')
    print(lookup([-1, 4, 1]))
    for name in categorical_feature_names:
        vocab = sorted(set(df[name]))
        print(f'name: {name}')
        print(f'vocab: {vocab}\n')

        if type(vocab[0]) is str:
            lookup = keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
        else:
            lookup = keras.layers.IntegerLookup(vocabulary=vocab, output_mode='one_hot')

        x = inputs[name]
        x = lookup(x)
        preprocessed.append(x)
    print(preprocessed)
    preprocessed_result = keras.layers.Concatenate(axis=1)(preprocessed)
    print(preprocessed_result)
    preprocessor = keras.Model(inputs, preprocessed_result)
    keras.utils.plot_model(preprocessor, to_file="preprocessor.png", rankdir="LR", show_shapes=True, show_layer_names=True)
    print("preprocessor")
    print(preprocessor(dict(df.iloc[:1])))

    # create and train a model
    body = keras.Sequential([
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(10, activation='relu'),
        keras.layers.Dense(1)
    ])
    print(inputs)
    x = preprocessor(inputs)
    print(x)
    result = body(x)
    print(result)
    model = keras.Model(inputs, result)

    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
    history = model.fit(dict(df), target, epochs=5, batch_size=BATCH_SIZE)

    # using tf.data
    ds = tf.data.Dataset.from_tensor_slices((
        dict(df),
        target
    ))

    ds = ds.batch(BATCH_SIZE)
    import pprint

    for x, y in ds.take(1):
        pprint.pprint(x)
        print()
        print(y)

    history = model.fit(ds, epochs=5)
