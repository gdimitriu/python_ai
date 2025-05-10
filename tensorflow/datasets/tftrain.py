# https://www.tensorflow.org/tutorials/load_data/tfrecord
import IPython
import tensorflow as tf
import numpy as np
import keras
import os
import IPython.display as display


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@tf.py_function(Tout=tf.string)
def serialize_example(feature0, feature1, feature2, feature3):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def _parse_function(example_proto):
    global feature_description
    # Parse the input `tf.train.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


if __name__ == "__main__":
    global feature_description
    print(_bytes_feature(b'test_string'))
    print(_bytes_feature(u'test_bytes'.encode('utf-8')))

    print(_float_feature(np.exp(1)))

    print(_int64_feature(True))
    print(_int64_feature(1))
    feature = _float_feature(np.exp(1))

    print(feature.SerializeToString())

    # The number of observations in the dataset.
    n_observations = int(1e4)

    # Boolean feature, encoded as False or True.
    feature0 = np.random.choice([False, True], n_observations)

    # Integer feature, random from 0 to 4.
    feature1 = np.random.randint(0, 5, n_observations)

    # String feature.
    strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
    feature2 = strings[feature1]

    # Float feature, from a standard normal distribution.
    feature3 = np.random.randn(n_observations)

    # This is an example observation from the dataset.

    example_observation = [False, 4, b'goat', 0.9876]
    serialized_example = serialize_example(*example_observation)
    print(serialized_example)

    example_proto = tf.train.Example.FromString(serialized_example.numpy())
    print(example_proto)

    filename = 'tfrecord_tftrain/test.tfrecord'
    # Write the `tf.train.Example` observations to the file.
    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(n_observations):
            example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
            writer.write(example.numpy())

    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print(raw_dataset)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)

    result = {}
    # example.features.feature is the dictionary
    for key, feature in example.features.feature.items():
        # The values are the Feature objects which contain a `kind` which contains:
        # one of three fields: bytes_list, float_list, int64_list
        kind = feature.WhichOneof('kind')
        result[key] = np.array(getattr(feature, kind).value)

    print(result)

    # reading a tfrecord using tf.data
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print(raw_dataset)

    for raw_record in raw_dataset.take(10):
        print(repr(raw_record))

    # Create a description of the features.
    feature_description = {
        'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
    }

    parsed_dataset = raw_dataset.map(_parse_function)
    print(parsed_dataset)

    for parsed_record in parsed_dataset.take(10):
        print(repr(parsed_record))

    cat_in_snow = tf.keras.utils.get_file(
        '320px-Felis_catus-cat_on_snow.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')

    williamsburg_bridge = tf.keras.utils.get_file(
        '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')

    display.display(display.Image(filename=cat_in_snow))
    display.display(display.HTML(
        'Image cc-by: <a "href=https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg">Von.grzanka</a>'))
    display.display(display.Image(filename=williamsburg_bridge))
    display.display(display.HTML(
        '<a "href=https://commons.wikimedia.org/wiki/File:New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg">From Wikimedia</a>'))

    image_labels = {
        cat_in_snow: 0,
        williamsburg_bridge: 1,
    }
    # This is an example, just using the cat image.
    image_string = open(cat_in_snow, 'rb').read()

    label = image_labels[cat_in_snow]


    # Create a dictionary with features that may be relevant.
    def image_example(image_string, label):
        image_shape = tf.io.decode_jpeg(image_string).shape

        feature = {
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'depth': _int64_feature(image_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


    for line in str(image_example(image_string, label)).split('\n')[:15]:
        print(line)
    print('...')

    # Write the raw image files to `images.tfrecords`.
    # First, process the two images into `tf.train.Example` messages.
    # Then, write to a `.tfrecords` file.
    record_file = 'tfrecord_tftrain/images.tfrecords'
    with tf.io.TFRecordWriter(record_file) as writer:
        for filename, label in image_labels.items():
            image_string = open(filename, 'rb').read()
            tf_example = image_example(image_string, label)
            writer.write(tf_example.SerializeToString())

    raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

    # Create a dictionary describing the features.
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }


    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)


    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    print(parsed_image_dataset)

    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        display.display(display.Image(data=image_raw))
