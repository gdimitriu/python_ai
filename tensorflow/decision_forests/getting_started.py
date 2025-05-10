# https://www.tensorflow.org/decision_forests/tutorials/beginner_colab
import os
# Keep using Keras 2
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow_decision_forests as tfdf

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import math

# Check the version of TensorFlow Decision Forests
print("Found TensorFlow Decision Forests v" + tfdf.__version__)