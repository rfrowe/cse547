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


def get_dev_loss(sess, inp, label, data, num_batches, loss):
    _logger.info("Calculating dev loss...")

    loss_val = 0.
    for _ in tqdm(range(num_batches)):
        (x, _), y = sess.run(data)
        if not isinstance(y, np.ndarray):
            y = np.array([[y]])
        y = y.reshape((y.shape[-1], 1))

        loss_val += np.array(sess.run(fetches=loss, feed_dict={inp: x, label: y}))

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
    encoder = Encoder(0)
    inp = tf.image.per_image_standardization(tf.placeholder(tf.float32, shape=[batch_size, *shape]))
    out = encoder(inp)
    out = tf.reshape(out, [int(inp.shape[0]), reduce(operator.mul, out.shape[1:])])

    saver = tf.train.Saver(var_list=encoder.variables)
    saver.restore(sess, weights_path)

    return encoder, (inp, out)
