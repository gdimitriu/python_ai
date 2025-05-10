# https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing

import tensorflow as tf
import keras
import numpy as np


class Linear(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Linear_bestPractice(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class Linear_build(keras.layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.total = self.add_weight(
            initializer="zeros", shape=(input_dim,), trainable=False
        )

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.linear_1 = Linear_build(32)
        self.linear_2 = Linear_build(32)
        self.linear_3 = Linear_build(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_mean(inputs))
        return inputs


class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


if __name__ == "__main__":
    x = tf.ones((2, 2))
    linear_layer = Linear(4, 2)
    y = linear_layer(x)
    print(y)
    assert linear_layer.weights == [linear_layer.w, linear_layer.b]
    x = tf.ones((2, 2))
    my_sum = ComputeSum(2)
    y = my_sum(x)
    print(y.numpy())
    y = my_sum(x)
    print(y.numpy())

    print("weights:", len(my_sum.weights))
    print("non-trainable weights:", len(my_sum.non_trainable_weights))

    # It's not included in the trainable weights:
    print("trainable_weights:", my_sum.trainable_weights)

    # At instantiation, we don't know on what inputs this is going to get called
    linear_layer = Linear_build(32)

    # The layer's weights are created dynamically the first time the layer is called
    y = linear_layer(x)

    mlp = MLPBlock()
    y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
    print("weights:", len(mlp.weights))
    print("trainable weights:", len(mlp.trainable_weights))

    layer = OuterLayer()
    assert len(layer.losses) == 0  # No losses yet since the layer has never been called

    _ = layer(tf.zeros(1, 1))
    assert len(layer.losses) == 1  # We created one loss value

    # `layer.losses` gets reset at the start of each __call__
    _ = layer(tf.zeros(1, 1))
    assert len(layer.losses) == 1  # This is the loss created during the call above

    layer = OuterLayerWithKernelRegularizer()
    _ = layer(tf.zeros((1, 1)))

    # This is `1e-3 * sum(layer.dense.kernel ** 2)`,
    # created by the `kernel_regularizer` above.
    print(layer.losses)

    # Instantiate an optimizer.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    inputs = keras.Input(shape=(3,))
    outputs = ActivityRegularizationLayer()(inputs)
    model = keras.Model(inputs, outputs)

    # If there is a loss passed in `compile`, the regularization
    # losses get added to it
    model.compile(optimizer="adam", loss="mse")
    model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

    # It's also possible not to pass any loss in `compile`,
    # since the model already has a loss to minimize, via the `add_loss`
    # call during the forward pass!
    model.compile(optimizer="adam")
    model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
