"""
Defines 2-d autoencoder in TensorFlow.
"""
import operator
from functools import reduce

from tensorflow.contrib.keras import layers as _layers
import tensorflow as tf


class Encoder(_layers.Layer):
    def __init__(self, hidden_dims):
        super().__init__()
        self._hidden_dims = hidden_dims

        self._conv1 = _layers.Conv2D(32, 5, strides=2, padding='same', activation=tf.nn.leaky_relu, name="conv1")
        self._pool1 = _layers.MaxPool2D(strides=2, padding='same', name="pool1")
        self._conv2 = _layers.Conv2D(64, 5, strides=2, padding='same', activation=tf.nn.leaky_relu, name="conv2")
        self._pool2 = _layers.MaxPool2D(strides=2, padding='same', name="pool2")
        self._dropout1 = _layers.Dropout(0.1)
        self._dense1 = _layers.Dense(hidden_dims, activation=tf.nn.leaky_relu)

    def call(self, tensor, **kwargs):
        output = self._conv1(tensor)
        output = self._pool1(output)
        output = self._conv2(output)
        output = self._pool2(output)
        output = self._dropout1(output)

        flat = reduce(operator.mul, output.shape[1:])
        output = tf.reshape(output, (-1, flat))
        output = self._dense1(output)
        return output


class Decoder(_layers.Layer):
    def __init__(self, hidden_dims):
        super().__init__()
        self._hidden_dims = hidden_dims

        self._conv1 = _layers.Conv2DTranspose(64, 5, strides=2, padding='same', name="conv1")
        self._conv2 = _layers.Conv2DTranspose(32, 5, strides=2, padding='same', name="conv2")
        self._conv3 = _layers.Conv2DTranspose(16, 5, strides=2, padding='same', name="conv3")
        self._conv4 = _layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation=tf.nn.tanh, name="conv4")

    def call(self, tensor, **kwargs):
        tensor = tf.reshape(tensor, (-1, 16, 16, 1))
        output = self._conv1(tensor)
        output = self._conv2(output)
        output = self._conv3(output)
        output = self._conv4(output)
        return output


class Autoencoder(_layers.Layer):
    def __init__(self, hidden_dims):
        super().__init__()
        self._encoder = Encoder(hidden_dims)
        self._decoder = Decoder(hidden_dims)

    def call(self, tensor, **kwargs):
        latent = self._encoder(tensor)
        recons = self._decoder(latent)
        return recons
