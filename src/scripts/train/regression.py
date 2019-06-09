#!/usr/bin/env python3
"""
Regression model using autoencoder to predict fluid intelligence.
"""

import operator
import os
from functools import reduce
from hashlib import md5

import tensorboard_logger as _tboard
import tensorflow as tf

import data.dataset as _dataset
import model.svr as _svr
import model.reco as _reco
import utils.cmd_line as _cmd
import utils.utility as _util
from model.autoencoder import Encoder

_logger = _util.get_logger(__file__)


def regression(dataset: str, encoder_weights: str, lr: float, epsilon: float, model: str):
    """
    Creates and trains a regression model with variable batch size.

    :param dataset: Name of dataset over which to train.
    :param encoder_weights: Path to trained encoder weights.
    :param lr:  Model learning rate.
    :param epsilon: Cutoff for training termination.
    :param model: Model type to use for regression
    """
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(encoder_weights, str) and len(encoder_weights)
    assert isinstance(lr, float) and lr > 0
    assert isinstance(epsilon, float) and epsilon > 0
    assert isinstance(model, str) and len(model)

    model = _get_model(model)

    if not os.path.isabs(encoder_weights):
        encoder_weights = _util.get_rel_weights_path(encoder_weights)

    # Note: these are weights for THIS model.
    weights_path = _util.get_weights_path_by_param(
        dataset="{}_predict".format(dataset),
        encoder=md5(encoder_weights.encode("ascii")).hexdigest(),
        lr=lr,
        epsilon=epsilon,
    )
    log_path = os.path.join(weights_path, "logs")

    train_dataset = _dataset.get_dataset_by_name(os.path.join(dataset, "train"), partial=True).batch(1)
    dev_dataset = _dataset.get_dataset_by_name(os.path.join(dataset, "dev"), partial=True).batch(1)
    shape = _dataset.load_shape(_util.get_rel_datasets_path(dataset))

    label = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    _tboard.configure(log_path, flush_secs=2)
    with tf.Session() as sess:
        # Define input, output, and intermediate operation.
        encoder, (inp, code) = _load_encoder(sess, encoder_weights, shape)

        model(sess, inp, code, label, epsilon, train_dataset, dev_dataset, lr, weights_path)


def _get_model(model: str) -> callable:
    model = model.lower()
    if model == "svr":
        return _svr.svr
    elif model == "reco":
        return _reco.reco
    else:
        # TODO: add more regression models.
        raise NotImplementedError("Model '{}' not implemented".format(model))


def _load_encoder(sess: tf.Session, weights_path: str, shape):
    encoder = Encoder(0)
    inp = tf.image.per_image_standardization(tf.placeholder(tf.float32, shape=[1, *shape]))
    out = encoder(inp)
    out = tf.reshape(out, [int(inp.shape[0]), reduce(operator.mul, out.shape[1:])])

    saver = tf.train.Saver(var_list=encoder.variables)
    saver.restore(sess, weights_path)

    return encoder, (inp, out)


def main():
    args = _cmd.parse_args_for_callable(regression)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)

    regression(**varsArgs)


if __name__ == "__main__":
    main()
