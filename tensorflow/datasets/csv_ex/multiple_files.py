# https://www.tensorflow.org/tutorials/load_data/csv
import pathlib
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
import keras


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Kmeans \
            using SalesTransaction data')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


def make_font_csv_ds(path):
  return tf.data.experimental.CsvDataset(
    path,
    record_defaults=font_column_types,
    header=True)


if __name__ == "__main__":
    # Make numpy values easier to read.
    np.set_printoptions(precision=3, suppress=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    args = build_arg_parser().parse_args()
    font_csvs = sorted(str(p) for p in pathlib.Path(args.input_dir, 'fonts').glob("*.csv"))
    print(font_csvs[:10])
    print(len(font_csvs))

    fonts_ds = tf.data.experimental.make_csv_dataset(
        file_pattern=args.input_dir + "/fonts/*.csv",
        batch_size=10, num_epochs=1,
        num_parallel_reads=20,
        shuffle_buffer_size=10000)
    for features in fonts_ds.take(1):
        for i, (name, value) in enumerate(features.items()):
            if i > 15:
                break
            print(f"{name:20s}: {value}")
    print('...')
    print(f"[total: {len(features)} features]")

    print("using low level")
    font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
    print(font_line)
    num_font_features = font_line.count(',') + 1
    font_column_types = [str(), str()] + [float()] * (num_font_features - 2)
    print(font_csvs[0])
    simple_font_ds = tf.data.experimental.CsvDataset(
        font_csvs,
        record_defaults=font_column_types,
        header=True)
    for row in simple_font_ds.take(10):
        print(row[0].numpy())

    font_files = tf.data.Dataset.list_files(args.input_dir + "/fonts/*.csv")
    print('Epoch 1:')
    for f in list(font_files)[:5]:
        print("    ", f.numpy())
    print('    ...')
    print()

    print('Epoch 2:')
    for f in list(font_files)[:5]:
        print("    ", f.numpy())
    print('    ...')
    font_rows = font_files.interleave(make_font_csv_ds, cycle_length=3)
    fonts_dict = {'font_name': [], 'character': []}

    for row in font_rows.take(10):
        fonts_dict['font_name'].append(row[0].numpy().decode())
        fonts_dict['character'].append(chr(int(row[2].numpy())))

    print(pd.DataFrame(fonts_dict))

