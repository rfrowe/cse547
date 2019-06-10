"""
Defines 3D convolutional autoencoder in TensorFlow.
"""
from tensorflow.contrib.keras import layers as _layers
from tensorflow.contrib.keras import regularizers as _regs
import tensorflow as tf


class Encoder(_layers.Layer):
    def __init__(self, l2_reg):
        super().__init__()

        reg = _regs.l2(l2_reg)
        initializer = tf.contrib.layers.xavier_initializer()
        self._conv1 = _layers.Conv3D(64, 3, strides=2, padding="same", name="conv1",
                                     activation=tf.nn.elu, kernel_regularizer=reg,
                                     kernel_initializer=initializer)
        self._conv2 = _layers.Conv3D(64, 3, strides=2, padding="same", name="conv2",
                                     activation=tf.nn.elu, kernel_regularizer=reg,
                                     kernel_initializer=initializer)
        self._conv3 = _layers.Conv3D(128, 3, strides=2, padding="same", name="conv3",
                                     activation=tf.nn.elu, kernel_regularizer=reg,
                                     kernel_initializer=initializer)
        self._conv4 = _layers.Conv3D(256, 3, strides=2, padding="same", name="conv4",
                                     activation=tf.nn.elu, kernel_regularizer=reg,
                                     kernel_initializer=initializer)
        self._conv5 = _layers.Conv3D(512, 3, strides=2, padding="same", name="conv5",
                                     activation=tf.nn.elu, kernel_regularizer=reg,
                                     kernel_initializer=initializer)
        self._conv6 = _layers.Conv3D(512, 3, strides=2, padding="same", name="conv6",
                                     activation=tf.nn.elu, kernel_regularizer=reg,
                                     kernel_initializer=initializer)

    def call(self, tensor, **kwargs):
        # Add channel dimension since these are monochrome.
        tensor = tf.expand_dims(tensor, -1)

        tensor = self._conv1(tensor)
        tensor = self._conv2(tensor)
        tensor = self._conv3(tensor)
        tensor = self._conv4(tensor)
        tensor = self._conv5(tensor)
        tensor = self._conv6(tensor)
        return tensor


class Decoder(_layers.Layer):
    def __init__(self):
        super().__init__()

        # TODO (rfrowe): is there a meaningful motivation to regularize decoder weights?
        # reg = _regs.l2(l2_reg)
        initializer = tf.contrib.layers.xavier_initializer()
        self._conv1 = _layers.Conv3DTranspose(512, 3, strides=2, padding="same", name="conv1",
                                              activation=tf.nn.elu,  # kernel_regularizer=reg,
                                              kernel_initializer=initializer)
        self._conv2 = _layers.Conv3DTranspose(256, 3, strides=2, padding="same", name="conv2",
                                              activation=tf.nn.elu,  # kernel_regularizer=reg,
                                              kernel_initializer=initializer)
        self._conv3 = _layers.Conv3DTranspose(128, 3, strides=2, padding="same", name="conv3",
                                              activation=tf.nn.elu,  # kernel_regularizer=reg,
                                              kernel_initializer=initializer)
        self._conv4 = _layers.Conv3DTranspose(64, 3, strides=2, padding="same", name="conv4",
                                              activation=tf.nn.elu,  # kernel_regularizer=reg,
                                              kernel_initializer=initializer)
        self._conv5 = _layers.Conv3DTranspose(64, 3, strides=2, padding="same", name="conv5",
                                              activation=tf.nn.elu,  # kernel_regularizer=reg,
                                              kernel_initializer=initializer)
        self._conv6 = _layers.Conv3DTranspose(1, 3, strides=2, padding="same", name="conv6",
                                              activation=tf.nn.elu,  # kernel_regularizer=reg,
                                              kernel_initializer=initializer)

    def call(self, tensor, **kwargs):
        tensor = self._conv1(tensor)
        tensor = self._conv2(tensor)
        tensor = self._conv3(tensor)
        tensor = self._conv4(tensor)
        tensor = self._conv5(tensor)
        tensor = self._conv6(tensor)
        return tf.squeeze(tensor, 4)


class Autoencoder(_layers.Layer):
    def __init__(self, l2_reg):
        super().__init__()
        self._encoder = Encoder(l2_reg)
        self._decoder = Decoder()

    def call(self, tensor, **kwargs):
        latent = self._encoder(tensor)
        recons = self._decoder(latent)
        return recons
