"""
Regression model using autoencoder to predict fluid intelligence.
"""
import os
from _md5 import md5

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tensorboard_logger as _tboard

import utils.cmd_line as _cmd
import utils.utility as _util

import data.dataset as _dataset

from model.autoencoder import Encoder
from train.autoencoder import _iter_len

_logger = _util.get_logger(__file__)


def svr(dataset: str, encoder_weights: str, batch_size: int, buffer_size: int, lr: float, epsilon: float):
    """
    Creates and trains an SVR model with variable batch size.

    TODO: describe params
    TODO: kernels?
    :param dataset:
    :param encoder_weights:
    :param batch_size:
    :param buffer_size:
    :param lr:
    :param epsilon:
    :return:
    """
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(encoder_weights, str) and len(encoder_weights)
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(buffer_size, int) and buffer_size > 0
    assert isinstance(lr, float) and lr > 0
    assert isinstance(epsilon, float) and epsilon > 0

    # Ensure weights path exists.
    if not os.path.isabs(encoder_weights):
        encoder_weights = _util.get_rel_weights_path(encoder_weights)
    _util.ensure_dir(encoder_weights)

    # Note: these are weights for THIS model.
    weights_path = _util.get_weights_path_by_param(
        dataset="{}_predict".format(dataset),
        encoder=md5(encoder_weights),
        batch_size=batch_size,
        buffer_size=buffer_size,
        lr=lr,
        epsilon=epsilon,
    )
    log_path = os.path.join(weights_path, "logs")

    train_dataset = _dataset.get_dataset_by_name(os.path.join(dataset, "train"), partial=True)
    dev_dataset = _dataset.get_dataset_by_name(os.path.join(dataset, "dev"), partial=True)

    label = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    _tboard.configure(log_path, flush_secs=2)
    with tf.Session() as sess:
        # Define input, output, and intermediate operation.
        encoder, (inp, code) = _load_encoder(sess, weights_path, shape)

        w = tf.Variable(tf.random_normal(shape=[1, code.shape[-1]]))
        b = tf.Variable(tf.random_normal(shape=[1]))
        saver = tf.train.Saver(var_list=[w, b])

        out = tf.add(tf.matmul(code, w), b)

        # Define loss function and get ready for training.
        loss = tf.reduce_mean(tf.maximum(tf.abs(out - label) + epsilon)) + tf.nn.l2_loss(w)
        train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
        sess.run(tf.global_variables_initializer())

        _logger.info("Counting datasets...")
        train_batches = _iter_len(sess, train_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tTrain samples: {}".format(train_batches))
        dev_batches = _iter_len(sess, dev_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tDev samples: {}".format(dev_batches))

        losscrit = np.inf
        dev_losses = []
        epoch = 0
        while losscrit > 1e-5:
            train_iter = train_dataset.make_one_shot_iterator().get_next()

            train_loss = 0.
            for _ in range(train_batches):
                (x, _), y = sess.run(train_iter)

                _, _train_loss = sess.run(fetches=[train_step, loss], feed_dict={inp: x, label: y})
                train_loss += _train_loss

            dev_loss = _get_dev_loss(sess, inp, dev_dataset, dev_batches, loss)

            _logger.info("Epoch {}: train {} dev {}".format(epoch, train_loss, dev_loss))
            _tboard.log_value("train", train_loss, step=epoch)
            _tboard.log_value("dev", dev_loss, step=epoch)
            if dev_loss < min(dev_losses):
                save_path = saver.save(sess, os.path.join(weights_path, "{}.ckpt".format(epoch)))
                _logger.info("Saved new best model to {}".format(save_path))
            dev_losses.append(dev_loss)

            if len(dev_losses) > 1:
                losscrit = (dev_loss[-1] - dev_loss[-2]) / dev_loss[-2]
            _tboard.log_value("losscrit", losscrit, step=epoch)


def _get_dev_loss(sess, inp, data, num_batches, loss):
    data = data.make_one_shot_iterator().get_next()

    _logger.info("Calculating dev loss...")

    loss_val = 0.
    for _ in tqdm(range(num_batches)):
        loss_val += np.array(sess.run(fetches=loss, feed_dict={inp: sess.run(data)}))

    return loss_val


def _load_encoder(sess: tf.Session, weights_path: str, shape):
    encoder = Encoder(0)
    inp = tf.image.per_image_standardization(tf.placeholder(tf.float32, shape=[None, *shape]))
    out = tf.layers.Flatten(encoder(inp))

    saver = tf.train.Saver(var_list=encoder.variables)
    saver.restore(sess, weights_path)

    return encoder, (inp, out)


def main():
    args = _cmd.parse_args_for_callable(svr)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)

    svr(**varsArgs)
