"""
Defines 2-d autoencoder in TensorFlow.
"""
import pywt

from tensorflow.contrib.keras import layers as _layers
from tensorflow.contrib.keras import regularizers as _regs
import tensorflow as tf
import numpy as np


def _dwt_kernel_init(kind="bior2.2"):
    wavelet = pywt.Wavelet(kind)
    dec_hi = np.array(wavelet.dec_hi[::-1])
    dec_lo = np.array(wavelet.dec_lo[::-1])

    filters = np.array([
        dec_lo[None, :] * dec_lo[:, None],
        dec_lo[None, :] * dec_hi[:, None],
        dec_hi[None, :] * dec_lo[:, None],
        dec_hi[None, :] * dec_hi[:, None]
    ])

    return filters


def _idwt_kernel_init(kind="bior2.2"):
    wavelet = pywt.Wavelet(kind)
    dec_hi = np.array(wavelet.dec_hi)
    dec_lo = np.array(wavelet.dec_lo)

    filters = np.array([
        dec_lo[None, :] * dec_lo[:, None],
        dec_lo[None, :] * dec_hi[:, None],
        dec_hi[None, :] * dec_lo[:, None],
        dec_hi[None, :] * dec_hi[:, None]
    ])

    return filters


def _l2(reg):
    return _regs.l2(reg)


class Encoder(_layers.Layer):
    def __init__(self, l2_reg):
        super().__init__()

        self._conv1 = _layers.Conv3D(32, 5, strides=2, padding="same", activation=tf.nn.leaky_relu, name="conv1", kernel_regularizer=_l2(l2_reg))
        self._conv2 = _layers.Conv3D(256, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv2", kernel_regularizer=_l2(l2_reg))
        self._conv3 = _layers.Conv3D(1024, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv3", kernel_regularizer=_l2(l2_reg))
        self._conv4 = _layers.Conv3D(4096, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv4", kernel_regularizer=_l2(l2_reg))

    def call(self, tensor, **kwargs):
        # Add channel dimension since these are monochrome.
        tensor = tf.expand_dims(tensor, -1)

        output = self._conv1(tensor)
        output = self._conv2(output)
        output = self._conv3(output)
        output = self._conv4(output)
        return output


class Decoder(_layers.Layer):
    def __init__(self, l2_reg):
        super().__init__()

        # self._conv1 = _layers.Conv3DTranspose(4096, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv1", kernel_regularizer=_l2(l2_reg))
        self._conv2 = _layers.Conv3DTranspose(1024, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv2", kernel_regularizer=_l2(l2_reg))
        self._conv3 = _layers.Conv3DTranspose(256, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv3", kernel_regularizer=_l2(l2_reg))
        self._conv4 = _layers.Conv3DTranspose(32, 5, strides=4, padding="same", activation=tf.nn.leaky_relu, name="conv4", kernel_regularizer=_l2(l2_reg))
        self._conv5 = _layers.Conv3DTranspose(1, 5, strides=2, padding="same", activation=tf.nn.leaky_relu, name="conv5", kernel_regularizer=_l2(l2_reg))

    def call(self, tensor, **kwargs):
        # output = self._conv1(tensor)
        output = self._conv2(tensor)
        output = self._conv3(output)
        output = self._conv4(output)
        output = self._conv5(output)
        return tf.squeeze(output, 4)


class Autoencoder(_layers.Layer):
    def __init__(self, l2_reg):
        super().__init__()
        self._encoder = Encoder(l2_reg)
        self._decoder = Decoder(l2_reg)

    def call(self, tensor, **kwargs):
        latent = self._encoder(tensor)
        recons = self._decoder(latent)
        return recons
