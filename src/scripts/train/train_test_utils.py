import operator
from functools import reduce

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import List

import utils.utility as _util


from data.hcp_config import SUBJECTS
from model.autoencoder import Encoder

_logger = _util.get_logger(__file__)


def get_dev_loss(sess, scan, features, label, data, num_batches, loss):
    _logger.info("Calculating dev loss...")

    loss_val = 0.
    for _ in tqdm(range(num_batches)):
        (s, f), y = sess.run(data)
        if not isinstance(y, np.ndarray):
            y = np.array([[y]])
        y = y.reshape((y.shape[-1], 1))

        loss_val += np.array(sess.run(fetches=loss, feed_dict={scan: s, features: f, label: y}))

    return loss_val


def dataset_iter_len(sess, data):
    count = 0
    for _ in tqdm(SUBJECTS):
        try:
            sess.run(data)
            count += 1
        except tf.errors.OutOfRangeError:
            return count
    return count


def load_encoder(sess: tf.Session, weights_path: str, batch_size: int, shape: List[int]):
    encoder = Encoder(1)
    inp = tf.image.per_image_standardization(tf.placeholder(tf.float32, shape=[batch_size, *shape]))
    out = encoder(inp)
    out = tf.reshape(out, [int(inp.shape[0]), reduce(operator.mul, out.shape[1:])])

    saver = tf.train.Saver(var_list=encoder.variables)
    saver.restore(sess, weights_path)

    return encoder, (inp, out)


def print_error_metrics(sess, scan, features, out, label, data, data_len):
    mse = tf.reduce_mean(tf.squared_difference(label, out))
    mae = tf.reduce_mean(tf.abs(label - out))

    preds = []
    labels = []
    mses = []
    maes = []

    _logger.info("Computing error metrics...")

    for batch in tqdm(range(data_len)):
        (s, f), y = sess.run(data)
        if y.ndim < 2:
            y = y.reshape((y.shape[-1], 1))

        _pred, _mse, _mae = sess.run(fetches=[out, mse, mae], feed_dict={scan: s, features: f, label: y})
        _logger.info("Batch {} pred {} truth {} MSE {} MAE {}".format(batch + 1, _pred, y, _mse, _mae))
        preds.append(_pred)
        labels.append(y)
        mses.append(_mse)
        maes.append(_mae)

    preds = np.array(preds)
    labels = np.array(labels)
    mses = np.array(mses)
    maes = np.array(maes)

    mvar = np.square(labels - labels.mean()).mean()
    R = 1 - mses.mean() / mvar
    mape = 100 * (np.abs(labels - preds) / preds).mean()
    _logger.info("Test Avg Pred {} MSE {} MAE {} R^2 {} MAPE {}".format(preds.mean(), mses.mean(), maes.mean(), R, mape))
