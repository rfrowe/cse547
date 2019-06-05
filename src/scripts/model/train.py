#/usr/bin/env python3
import os

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tqdm import tqdm

import utils.cmd_line as _cmd
import utils.utility as _util

from model.autoencoder import Autoencoder
from data.dataset import get_dataset, load_shape

_logger = _util.get_logger(__file__)


def train(dataset: str, weights: str, epochs=100, batch_size=64, grad_norm=1000, buffer_size=8, lr=1e-3, l2_reg=1e-1, tv_reg=1e-2, partial=False):
    """
    TODO (rfrowe)
    :param dataset: 
    :param weights: 
    :param epochs: 
    :param batch_size: 
    :param grad_norm:
    :param buffer_size: 
    :param lr: 
    :param partial: 
    :return: 
    """""
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(weights, str) and len(weights)
    assert isinstance(epochs, int) and epochs > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(grad_norm, int) and grad_norm > 0
    assert isinstance(buffer_size, int) and batch_size > 0
    assert isinstance(lr, float) and lr > 0
    assert isinstance(l2_reg, float) and l2_reg > 0
    assert isinstance(tv_reg, float) and tv_reg > 0
    assert isinstance(partial, bool)

    model = Autoencoder(l2_reg)
    model_path = _util.get_rel_weights_path(model)
    _util.ensure_path_free(model_path)
    _util.mkdir(model_path)

    dataset_path = _util.get_rel_datasets_path(dataset)
    _util.ensure_dir(dataset_path)
    shape = load_shape(dataset_path)

    input = tf.placeholder(tf.float32, shape=[None, *shape])

    output = model.call(input)

    loss = tf.nn.l2_loss(input - output)
    if l2_reg > 0:
        loss += tf.add_n(model.losses)
    loss += tf.reduce_sum(_total_variation_5d(tf.expand_dims(output, 4)))

    op = tf.train.AdamOptimizer(learning_rate=lr)

    if grad_norm > 0:
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tvars, grad_norm)
        grads_and_vars = zip(grads, tvars)
    else:
        grads_and_vars = op.compute_gradients(loss)
    train_op = op.apply_gradients(grads_and_vars)

    train_dataset = _get_dataset(dataset, batch_size, buffer_size, partial)
    eval_dataset = _get_dataset(dataset, batch_size, buffer_size, partial)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    best_eval = np.inf
    with tf.Session() as sess:
        sess.run(init)

        _logger.info("Counting datasets...")
        train_samples = _iter_len(sess, train_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tTrain samples: {}".format(train_samples))
        eval_samples = _iter_len(sess, eval_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tEval samples: {}".format(eval_samples))

        # print("Epoch 0: eval {}".format(_get_eval_loss(sess, input, eval_dataset, eval_samples, loss)))
        for epoch in tqdm(range(epochs)):
            train_iter = train_dataset.make_one_shot_iterator().get_next()

            # train_loss = 0
            for batch in range(train_samples):

                _ = sess.run([train_op], feed_dict={input: train_iter})
                # train_loss += batch_loss
                # print("\tBatch {}: train {}".format(batch + 1, train_loss / ((batch + 1) / train_samples)))

            eval_loss = _get_eval_loss(sess, input, eval_dataset, eval_samples, loss)
            print("Epoch {}: eval {}".format(epoch + 1, eval_loss))
            if eval_loss < best_eval:
                save_path = saver.save(sess, os.path.join(model_path, "{}.ckpt".format(epochs + 1)))
                _logger.info("Saved new best model to {}".format(save_path))
                best_eval = eval_loss

            epoch += 1


def _iter_len(sess, iter):
    count = 0
    while True:
        try:
            sess.run(iter)
            count += 1
        except tf.errors.OutOfRangeError:
            return count


def _get_eval_loss(sess, input, data, data_len, loss):
    data = data.make_one_shot_iterator().get_next()

    value = 0.
    for _ in range(data_len):
        value += sess.run(fetches=loss, feed_dict={input: data})

    return value / data_len


def _get_dataset(name, batch_size, buffer_size, partial):
    dataset_path = _util.get_rel_datasets_path(name)
    _util.ensure_dir(dataset_path)

    return get_dataset(dataset_path, batch_size=batch_size, buffer_size=buffer_size, partial=partial).map(_only_cropped_scan)


def _only_cropped_scan(*data):
    return data[0][0]


def _total_variation_5d(images, name=None):
    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims

        if ndims == 5:
            # The input is a batch of images with shape:
            # [batch, height, width, depth, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = images[:, 1:, :, :, :] - images[:, :-1, :, :, :]
            pixel_dif2 = images[:, :, 1:, :, :] - images[:, :, :-1, :, :]
            pixel_dif3 = images[:, :, :, 1:, :] - images[:, :, :, :-1, :]

            # Only sum for the last 4 axis.
            # This results in a 1-D tensor with the total variation for each image.
            sum_axis = [1, 2, 3, 4]
        else:
            raise ValueError('\'images\' must be 5-dimensional.')

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences and summing over the appropriate axis.
        tot_var = (
                math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
                math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis) +
                math_ops.reduce_sum(math_ops.abs(pixel_dif3), axis=sum_axis))

    return tot_var


def main():
    args = _cmd.parse_args_for_callable(train)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    train(**varsArgs)


if __name__ == '__main__':
    main()

