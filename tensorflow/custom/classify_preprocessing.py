# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from keras import layers

def build_arg_parser():
    parser = argparse.ArgumentParser(description='classify preprocessing')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('target')
    df = {key: value.to_numpy()[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds


def get_normalization_layer(name, dataset):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    # Create a layer that turns strings into integer indices.
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    args = build_arg_parser().parse_args()
    # dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    # csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'
    csv_file = args.input_dir + "/petfinder-mini/petfinder-mini.csv"

    # tf.keras.utils.get_file('petfinder_mini.zip', dataset_url, extract=True, cache_dir='.')
    dataframe = pd.read_csv(csv_file)
    print(dataframe.head())
    # In the original dataset, `'AdoptionSpeed'` of `4` indicates
    # a pet was not adopted.
    dataframe['target'] = np.where(dataframe['AdoptionSpeed'] == 4, 0, 1)

    # Drop unused features.
    dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])
    train, val, test = np.split(dataframe.sample(frac=1), [int(0.8 * len(dataframe)), int(0.9 * len(dataframe))])
    print(len(train), 'training examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')
    batch_size = 5
    train_ds = df_to_dataset(train, batch_size=batch_size)
    [(train_features, label_batch)] = train_ds.take(1)
    print('Every feature:', list(train_features.keys()))
    print('A batch of ages:', train_features['Age'])
    print('A batch of targets:', label_batch)
    photo_count_col = train_features['PhotoAmt']
    layer = get_normalization_layer('PhotoAmt', train_ds)
    print(layer(photo_count_col))
    test_type_col = train_features['Type']
    test_type_layer = get_category_encoding_layer(name='Type',
                                                  dataset=train_ds,
                                                  dtype='string')
    print(test_type_layer(test_type_col))
    test_age_col = train_features['Age']
    test_age_layer = get_category_encoding_layer(name='Age',
                                                 dataset=train_ds,
                                                 dtype='int64',
                                                 max_tokens=5)
    print(test_age_layer(test_age_col))
    batch_size = 256
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)
    all_inputs = {}
    encoded_features = []

    # Numerical features.
    for header in ['PhotoAmt', 'Fee']:
        numeric_col = tf.keras.Input(shape=(1,), name=header)
        normalization_layer = get_normalization_layer(header, train_ds)
        encoded_numeric_col = normalization_layer(numeric_col)
        all_inputs[header] = numeric_col
        encoded_features.append(encoded_numeric_col)

    age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')

    encoding_layer = get_category_encoding_layer(name='Age',
                                                 dataset=train_ds,
                                                 dtype='int64',
                                                 max_tokens=5)
    encoded_age_col = encoding_layer(age_col)
    all_inputs['Age'] = age_col
    encoded_features.append(encoded_age_col)
    categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                        'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']

    for header in categorical_cols:
        categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
        encoding_layer = get_category_encoding_layer(name=header,
                                                     dataset=train_ds,
                                                     dtype='string',
                                                     max_tokens=5)
        encoded_categorical_col = encoding_layer(categorical_col)
        all_inputs[header] = categorical_col
        encoded_features.append(encoded_categorical_col)

    # Create, compile, and train the model
    print(encoded_features)
    all_features = tf.keras.layers.concatenate(encoded_features)
    x = tf.keras.layers.Dense(32, activation="relu")(all_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(all_inputs, output)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"],
                  run_eagerly=True)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"],
                  run_eagerly=True)
    model.fit(train_ds, epochs=10, validation_data=val_ds)
    result = model.evaluate(test_ds, return_dict=True)
    print(result)

    # performance inference
    model.save('my_pet_classifier.keras')
    reloaded_model = tf.keras.models.load_model('my_pet_classifier.keras')
    sample = {
        'Type': 'Cat',
        'Age': 3,
        'Breed1': 'Tabby',
        'Gender': 'Male',
        'Color1': 'Black',
        'Color2': 'White',
        'MaturitySize': 'Small',
        'FurLength': 'Short',
        'Vaccinated': 'No',
        'Sterilized': 'No',
        'Health': 'Healthy',
        'Fee': 100,
        'PhotoAmt': 2,
    }

    input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
    predictions = reloaded_model.predict(input_dict)
    prob = tf.nn.sigmoid(predictions[0])

    print(
        "This particular pet had a %.1f percent probability "
        "of getting adopted." % (100 * prob)
    )
