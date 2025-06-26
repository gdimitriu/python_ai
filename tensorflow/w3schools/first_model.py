import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

xs = tf.constant([0, 1, 2, 3, 4])
ys = tf.multiply(tf.cast(xs, dtype=tf.float64), 1.2)
ys = tf.add(ys, 5.0)
fig = plt.figure()
plt.scatter(xs, ys)
plt.show()
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=[1]))
model.compile(optimizer=keras.optimizers.SGD(), loss=keras.losses.MeanSquaredError)
x_train = np.array(xs)
y_train = np.array(ys)
model.fit(x_train, y_train, epochs=500)

xMax = 10
xArr = []
yArr = []
fig = plt.figure()
for x in range(xMax + 1):
    result = model.predict(np.array([x]))
    val = result[0]
    xArr.append(x)
    yArr.append(val)
plt.scatter(xArr,yArr)
plt.show()

