#/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import utils.cmd_line as _cmd
import utils.utility as _util

from model.autoencoder import Autoencoder
from data.dataset import get_dataset

_logger = _util.get_logger(__file__)


def train(dataset: str, epochs=100, hidden_units=256, batch_size=64, buffer_size=8, lr=1e-3, partial=False):
    """
    TODO (rfrowe)
    :param dataset: 
    :param epochs: 
    :param hidden_units: 
    :param batch_size: 
    :param lr: 
    :param partial: 
    :return: 
    """""
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(epochs, int) and epochs > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(buffer_size, int) and batch_size > 0
    assert isinstance(hidden_units, int) and hidden_units > 0

    model = Autoencoder(hidden_units)
    input = tf.placeholder(tf.float32, shape=[None, 256, 256, 311])

    output = model.call(input)

    loss = tf.reduce_mean(tf.squared_difference(input, output))
    step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    init = tf.global_variables_initializer()

    train_dataset = _get_dataset(dataset, batch_size, buffer_size, partial).repeat()
    next_train = train_dataset.make_one_shot_iterator().get_next()

    eval_dataset = _get_dataset(dataset, batch_size, buffer_size, partial)

    with tf.Session() as sess:
        sess.run(init)

        print("Epoch 0: eval {}".format(_get_eval_loss(sess, input, eval_dataset, loss)))
        for epoch in tqdm(range(epochs)):
            sample = sess.run(next_train)

            step.run(feed_dict={input: sample}, session=sess)

            eval_loss = _get_eval_loss(sess, input, eval_dataset, loss)
            print("Epoch {}: eval {}".format(epoch + 1, eval_loss))

            epoch += 1


def _get_eval_loss(sess, input, data, loss):
    data = data.make_one_shot_iterator().get_next()

    value = 0.
    count = 0
    while True:
        try:
            value += loss.eval(feed_dict={input: sess.run(data)})
            count += 1
        except tf.errors.OutOfRangeError:
            return value / count if count > 0 else np.inf


def _get_dataset(name, batch_size, buffer_size, partial):
    dataset_path = _util.get_rel_datasets_path(name)
    _util.ensure_dir(dataset_path)

    return get_dataset(dataset_path, batch_size=batch_size, buffer_size=buffer_size, partial=partial).map(_only_cropped_scan)


def _only_cropped_scan(*data):
    return tf.transpose(data[0][0], perm=[0, 1, 3, 2])[:, 2:258, 2:258, :]


def main():
    args = _cmd.parse_args_for_callable(train)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    train(**varsArgs)


if __name__ == '__main__':
    main()

