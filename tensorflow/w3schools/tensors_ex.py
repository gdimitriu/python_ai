import tensorflow as tf

# from arrays
myArr = [[1, 2, 3, 4]]
tensorA = tf.convert_to_tensor(myArr)
print(tensorA)
myArr = [[1, 2], [3, 4]]
tensorA = tf.convert_to_tensor(myArr)
print(tensorA)
myArr = [[1, 2], [3, 4], [5, 6]]
tensorA = tf.convert_to_tensor(myArr)
print(tensorA)
# using shape
myArr = [1, 2, 3, 4]
shape = [2, 2]
tensorA = tf.convert_to_tensor(myArr)
tensorA = tf.reshape(tensorA, shape=shape)
print(tensorA)
tensorA = tf.convert_to_tensor([1, 2, 3, 4])
tensorA = tf.reshape(tensorA, shape=[2, 2])
print(tensorA)
myArr = [[1, 2], [3, 4]]
shape = [2, 2]
tensorA = tf.convert_to_tensor(myArr)
tensorA = tf.reshape(tensorA, shape=shape)
print(tensorA)
print("shape =",tensorA.shape)
print("array = ", tensorA.numpy())
print("dtype = ", tensorA.dtype)

# operations
tensorA = tf.constant([[1, 2], [3, 4], [5, 6]])
tensorB = tf.constant([[1,-1], [2,-2], [3,-3]])
tensorNew = tf.add(tensorA, tensorB)
print(tensorNew)
tensorNew = tf.subtract(tensorA, tensorB)
print(tensorNew)
tensorNew = tf.multiply(tensorA, tensorB)
print(tensorNew)
tensorA = tf.constant([2, 4, 6, 8])
tensorB = tf.constant([1, 2, 2, 2])
tensorNew = tf.divide(tensorA, tensorB)
print(tensorNew)
tensorA = tf.constant([1, 2, 3, 4])
tensorNew = tf.square(tensorA)
print(tensorNew)