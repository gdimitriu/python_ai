# https://www.tensorflow.org/guide/keras/working_with_rnns
import numpy as np
import tensorflow as tf
import keras
from keras import layers


def embedding_1():
    model = keras.Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=1000, output_dim=64, input_shape=(None,)))
    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))
    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))
    model.summary()

    # OUTPUT AND STATES
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64, input_shape=(None,)))
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))
    # The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
    model.add(layers.SimpleRNN(128))
    model.add(layers.Dense(10))
    model.summary()


def embedding_2():
    encoder_vocab = 1000
    decoder_vocab = 2000
    encoder_input = layers.Input(shape=(None,))
    encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
        encoder_input
    )
    # Return states in addition to output
    output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
        encoder_embedded
    )
    encoder_state = [state_h, state_c]
    decoder_input = layers.Input(shape=(None,))
    decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
        decoder_input
    )
    # Pass the 2 states to a new LSTM layer, as initial state
    decoder_output = layers.LSTM(64, name="decoder")(
        decoder_embedded, initial_state=encoder_state
    )
    output = layers.Dense(10)(decoder_output)
    model = keras.Model([encoder_input, decoder_input], output)
    model.summary()


def cross_batch_statefulness():
    # Cross-batch statefulness
    paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

    lstm_layer = layers.LSTM(64, stateful=True)
    output = lstm_layer(paragraph1)
    output = lstm_layer(paragraph2)
    output = lstm_layer(paragraph3)

    # reset_states() will reset the cached state to the original initial_state.
    # If no initial_state was provided, zero-states will be used by default.
    lstm_layer.reset_states()


def reuse_state():
    # RNN State Reuse
    paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
    paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)
    lstm_layer = layers.LSTM(64, stateful=True)
    output = lstm_layer(paragraph1)
    output = lstm_layer(paragraph2)
    existing_state = lstm_layer.states
    new_lstm_layer = layers.LSTM(64)
    new_output = new_lstm_layer(paragraph3, initial_state=existing_state)


def bidirectional_rnn():
    # Bidirectional RNNs
    model = keras.Sequential()

    model.add(
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
    )
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(10))
    model.summary()


def performance_opt_mnist():
    # Performance optimization and CuDNN kernels
    # Using CuDNN kernels when available
    batch_size = 64
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # Each input sequence will be of size (28, 28) (height is treated like time).
    input_dim = 28

    units = 64
    output_size = 10  # labels are from 0 to 9

    # Build the RNN model
    def build_model(allow_cudnn_kernel=True):
        # CuDNN is only available at the layer level, and not at the cell level.
        # This means `LSTM(units)` will use the CuDNN kernel,
        # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
        if allow_cudnn_kernel:
            # The LSTM layer with default options uses CuDNN.
            lstm_layer = keras.layers.LSTM(units, input_shape=(None, input_dim))
        else:
            # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
            lstm_layer = keras.layers.RNN(
                keras.layers.LSTMCell(units), input_shape=(None, input_dim)
            )
        model = keras.models.Sequential(
            [
                lstm_layer,
                keras.layers.BatchNormalization(),
                keras.layers.Dense(output_size),
            ]
        )
        return model

    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    sample, sample_label = x_train[0], y_train[0]
    model = build_model(allow_cudnn_kernel=True)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd",
        metrics=["accuracy"],
    )
    model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
    )
    noncudnn_model = build_model(allow_cudnn_kernel=False)
    noncudnn_model.set_weights(model.get_weights())
    noncudnn_model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd",
        metrics=["accuracy"],
    )
    noncudnn_model.fit(
        x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=1
    )
    import matplotlib.pyplot as plt

    with tf.device("CPU:0"):
        cpu_model = build_model(allow_cudnn_kernel=True)
        cpu_model.set_weights(model.get_weights())
        result = tf.argmax(cpu_model.predict_on_batch(tf.expand_dims(sample, 0)), axis=1)
        print(
            "Predicted result is: %s, target result is: %s" % (result.numpy(), sample_label)
        )
        plt.imshow(sample, cmap=plt.get_cmap("gray"))
        plt.show()


@keras.saving.register_keras_serializable()
class NestedCell(keras.layers.Layer):
    def __init__(self, unit_1, unit_2, unit_3, **kwargs):
        self.unit_1 = unit_1
        self.unit_2 = unit_2
        self.unit_3 = unit_3
        self.state_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
        self.output_size = [tf.TensorShape([unit_1]), tf.TensorShape([unit_2, unit_3])]
        super().__init__(**kwargs)

    def build(self, input_shapes):
        # expect input_shape to contain 2 items, [(batch, i1), (batch, i2, i3)]
        i1 = input_shapes[0][1]
        i2 = input_shapes[1][1]
        i3 = input_shapes[1][2]

        self.kernel_1 = self.add_weight(
            shape=(i1, self.unit_1), initializer="uniform", name="kernel_1"
        )
        self.kernel_2_3 = self.add_weight(
            shape=(i2, i3, self.unit_2, self.unit_3),
            initializer="uniform",
            name="kernel_2_3",
        )

    def call(self, inputs, states):
        # inputs should be in [(batch, input_1), (batch, input_2, input_3)]
        # state should be in shape [(batch, unit_1), (batch, unit_2, unit_3)]
        input_1, input_2 = tf.nest.flatten(inputs)
        s1, s2 = states

        output_1 = tf.matmul(input_1, self.kernel_1)
        output_2_3 = tf.einsum("bij,ijkl->bkl", input_2, self.kernel_2_3)
        state_1 = s1 + output_1
        state_2_3 = s2 + output_2_3

        output = (output_1, output_2_3)
        new_states = (state_1, state_2_3)

        return output, new_states

    def get_config(self):
        return {"unit_1": self.unit_1, "unit_2": self.unit_2, "unit_3": self.unit_3}


if __name__ == "__main__":
 #   embedding_1()
 #   embedding_2()
 #   cross_batch_statefulness()
 #   reuse_state()
 #   bidirectional_rnn()
    #performance_opt_mnist()

    # RNNs with list/dict inputs, or nested inputs
    unit_1 = 10
    unit_2 = 20
    unit_3 = 30
    i0 = 32
    i1 = 32
    i2 = 64
    i3 = 32
    batch_size = 64
    num_batches = 10
    timestep = 50

    cell = NestedCell(unit_1, unit_2, unit_3)
    rnn = keras.layers.RNN(cell)
    input_0 = keras.Input((i0,))
    input_1 = keras.Input((i1,))
    input_2 = keras.Input((i2, i3))
    outputs = rnn((input_1, input_0, input_2))
    model = keras.models.Model([input_1, input_2], outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
    input_1_data = np.random.random((batch_size * num_batches, timestep, i1))
    input_2_data = np.random.random((batch_size * num_batches, timestep, i2, i3))
    target_1_data = np.random.random((batch_size * num_batches, unit_1))
    target_2_data = np.random.random((batch_size * num_batches, unit_2, unit_3))
    input_data = [input_1_data, input_2_data]
    target_data = [target_1_data, target_2_data]

    model.fit(input_data, target_data, batch_size=batch_size)
