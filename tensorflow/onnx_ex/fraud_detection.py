import argparse
import numpy as np
import pandas as pd
import datetime
from keras import models
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tf2onnx
import onnx
import pickle
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description='fraud detection')
    parser.add_argument('--input-dir', dest='input_dir', type=str,
                        default='.', help='Directory for storing data')
    return parser


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.expand_frame_repr', False)
    args = build_arg_parser().parse_args()
    # Set the input (X) and output (Y) data.
    # The only output data is whether it's fraudulent. All other fields are inputs to the model.

    feature_indexes = [
        1,  # distance_from_last_transaction
        2,  # ratio_to_median_purchase_price
        4,  # used_chip
        5,  # used_pin_number
        6,  # online_order
    ]

    label_indexes = [
        7  # fraud
    ]

    df = pd.read_csv(args.input_dir + '/train.csv')
    X_train = df.iloc[:, feature_indexes].values
    y_train = df.iloc[:, label_indexes].values

    df = pd.read_csv(args.input_dir + '/validate.csv')
    X_val = df.iloc[:, feature_indexes].values
    y_val = df.iloc[:, label_indexes].values

    df = pd.read_csv(args.input_dir + '/test.csv')
    X_test = df.iloc[:, feature_indexes].values
    y_test = df.iloc[:, label_indexes].values

    # Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier for the model to learn than random (and potentially large) values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    Path("artifact").mkdir(parents=True, exist_ok=True)
    with open("artifact/test_data.pkl", "wb") as handle:
        pickle.dump((X_test, y_test), handle)
    with open("artifact/scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())
    class_weights = {i: class_weights[i] for i in range(len(class_weights))}

    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu', input_dim=len(feature_indexes)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train the model and get performance
    import os
    import time

    start = time.time()
    epochs = 2
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=True,
        class_weight=class_weights
    )
    end = time.time()
    print(f"Training of model is complete. Took {end - start} seconds")
    import tensorflow as tf


    # Wrap the model in a `tf.function`
    @tf.function(input_signature=[tf.TensorSpec([None, X_train.shape[1]], tf.float32, name='dense_input')])
    def model_fn(x):
        return model(x)


    # Convert the Keras model to ONNX
    model_proto, _ = tf2onnx.convert.from_function(
        model_fn,
        input_signature=[tf.TensorSpec([None, X_train.shape[1]], tf.float32, name='dense_input')]
    )

    # Save the model as ONNX for easy use of ModelMesh
    os.makedirs("models/fraud/1", exist_ok=True)
    onnx.save(model_proto, "models/fraud/1/model.onnx")
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import numpy as np
    import pickle
    import onnxruntime as rt

    with open('artifact/scaler.pkl', 'rb') as handle:
        scaler = pickle.load(handle)
    with open('artifact/test_data.pkl', 'rb') as handle:
        (X_test, y_test) = pickle.load(handle)
    sess = rt.InferenceSession("models/fraud/1/model.onnx", providers=rt.get_available_providers())
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y_pred_temp = sess.run([output_name], {input_name: X_test.astype(np.float32)})
    y_pred_temp = np.asarray(np.squeeze(y_pred_temp[0]))
    threshold = 0.95
    y_pred = np.where(y_pred_temp > threshold, 1, 0)
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
    import numpy as np

    y_test_arr = y_test.squeeze()
    correct = np.equal(y_pred, y_test_arr).sum().item()
    acc = (correct / len(y_pred)) * 100
    precision = precision_score(y_test_arr, np.round(y_pred))
    recall = recall_score(y_test_arr, np.round(y_pred))

    print(f"Eval Metrics: \n Accuracy: {acc:>0.1f}%, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f} \n")

    c_matrix = confusion_matrix(y_test_arr, y_pred)
    ConfusionMatrixDisplay(c_matrix).plot()
    sally_transaction_details = [
        [0.3111400080477545,
         1.9459399775518593,
         1.0,
         0.0,
         0.0]
    ]

    prediction = sess.run([output_name], {input_name: scaler.transform(sally_transaction_details).astype(np.float32)})

    print("Is Sally's transaction predicted to be fraudulent? (true = YES, false = NO) ")
    print(np.squeeze(prediction) > threshold)

    print("How likely was Sally's transaction to be fraudulent? ")
    print("{:.5f}".format(100 * np.squeeze(prediction)) + "%")
