# https://www.kaggle.com/code/amyjang/tensorflow-mnist-cnn-tutorial/notebook
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras

print(keras.__version__)
print(tf.__version__)
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# take too long time
# sns.countplot(y_train)
# plt.show()

print(np.isnan(x_train).any())
print(np.isnan(x_test).any())

# normalize data
input_shape = (28, 28, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test / 255.0

# label encoding
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

# visualize data
plt.imshow(x_train[100][:, :, 0])
print(y_train[100])
plt.show()
# CNN
batch_size = 64
num_classes = 10
epochs = 5

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    keras.layers.MaxPool2D(),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPool2D(strides=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=keras.optimizers.RMSprop(epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc') > 0.995):
            print("\nReached 99.5% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

x_train = np.array(x_train)
y_train = np.array(y_train)
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
                    callbacks=[callbacks])

fig, ax = plt.subplots(2, 1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)

x_test = np.array(x_test)
y_test = np.array(y_test)
test_loss, test_acc = model.evaluate(x_test, y_test)

# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert testing observations to one hot vectors
Y_true = np.argmax(y_test, axis=1)
# compute the confusion matrix
confusion_mtx = tf.math.confusion_matrix(Y_true, Y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g')
plt.show()
import sys

sys.stdin.read(1)
