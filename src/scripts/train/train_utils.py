import numpy as np
import tensorflow as tf
from tqdm import tqdm

import utils.utility as _util


from data.hcp_config import SUBJECTS

_logger = _util.get_logger(__file__)


def get_dev_loss(sess, inp, label, data, num_batches, loss):
    data = data.make_one_shot_iterator().get_next()

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