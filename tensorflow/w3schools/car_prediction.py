import json
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

def load_json():
    with open("carsData.json", "r", encoding="UTF-8") as source:
        objects = json.load(source)
    # print(objects)
    return objects

if __name__ == "__main__":
    jsonObject = load_json()
    xArray = [] # Horsepower
    yArray = [] # MPG
    for car in jsonObject:
        #print(type(car["Horsepower"]), type(car["Miles_per_Gallon"]))
        if car["Horsepower"] is not None and car["Miles_per_Gallon"] is not None:
            # print(car["Horsepower"], car["Miles_per_Gallon"])
            xArray.append(int(car["Horsepower"]))
            yArray.append(float(car["Miles_per_Gallon"]))
    fig = plt.figure()
    plt.scatter(xArray, yArray)
    plt.show()
    # convert
    inputTensor = tf.convert_to_tensor(xArray)
    labelTensor = tf.convert_to_tensor(yArray)
    # normalize
    inputMin = tf.math.reduce_min(inputTensor)
    inputMax = tf.math.reduce_max(inputTensor)
    labelMin = tf.math.reduce_min(labelTensor)
    labelMax = tf.math.reduce_max(labelTensor)
    tmp = tf.subtract(inputTensor, inputMin)
    nmInputs = tf.divide(tmp, tf.subtract(inputMax, inputMin))
    tmp = tf.subtract(labelTensor, labelMin)
    nmLabels = tf.divide(tmp, tf.subtract(labelMax, labelMin))
    figNormalized = plt.figure()
    plt.scatter(nmInputs.numpy(), nmLabels.numpy())
    plt.show()
    model = keras.models.Sequential([
        keras.layers.Dense(units=1, input_shape=[1], use_bias=True),
        keras.layers.Dense(units=1, use_bias=True)
    ])
    model.compile(loss=keras.losses.MeanSquaredError, optimizer=keras.optimizers.SGD())
    print(model.summary())
    model.fit(nmInputs.numpy(), nmLabels.numpy(), epochs=500, batch_size=25)
    unX = tf.linspace(0, 1, 100)
    unY = model.predict(tf.reshape(unX, [100,1]))
    # unnormalize
    unNormunX = tf.add(tf.multiply(unX, tf.cast(tf.subtract(inputMax, inputMin), dtype=tf.float64)),
                       tf.cast(inputMin, dtype=tf.float64))
    unNormunY = tf.add(tf.multiply(unY, tf.cast(tf.subtract(labelMax, labelMin), dtype=tf.float64)),
                       tf.cast(labelMin, dtype=tf.float64))
    fig = plt.figure()
    plt.scatter(xArray, yArray, color='r')
    plt.scatter(unNormunX, unNormunY, color='b')
    plt.show()

