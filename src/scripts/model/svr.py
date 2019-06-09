import os

import numpy as np
import tensorboard_logger as _tboard
import tensorflow as tf
from tqdm import tqdm

import utils.utility as _util
import train.train_utils as _train_utils

_logger = _util.get_logger(__file__)


def svr(sess, inp, code, label, epsilon, train_dataset, dev_dataset, lr, weights_path):
    w = tf.Variable(tf.random_normal(shape=[1, int(code.shape[-1])]))
    b = tf.Variable(tf.random_normal(shape=[1]))
    saver = tf.train.Saver(var_list=[w, b])

    out = tf.add(tf.matmul(code, w, transpose_b=True), b)

    # Define loss function and get ready for training.
    loss = tf.reduce_mean(tf.maximum(tf.abs(out - label) + epsilon, 0.)) + tf.nn.l2_loss(w)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    sess.run(tf.global_variables_initializer())

    _logger.info("Counting datasets...")
    train_batches = _train_utils.dataset_iter_len(sess, train_dataset.make_one_shot_iterator().get_next())
    _logger.info("\tTrain samples: {}".format(train_batches))
    dev_batches = _train_utils.dataset_iter_len(sess, dev_dataset.make_one_shot_iterator().get_next())
    _logger.info("\tDev samples: {}".format(dev_batches))

    losscrit = np.inf
    dev_losses = [np.inf]
    epoch = 0
    while losscrit > 1e-5:
        train_iter = train_dataset.shuffle(int(1.5 * train_batches)).make_one_shot_iterator().get_next()

        train_loss = 0.
        for b in tqdm(range(train_batches)):
            (x, _), y = sess.run(train_iter)
            y = y.reshape((y.shape[-1], 1))

            _, _train_loss = sess.run(fetches=[train_step, loss], feed_dict={inp: x, label: y})
            _tboard.log_value("loss/batch", _train_loss, step=epoch * train_batches + b)
            train_loss += _train_loss

        dev_loss = _train_utils.get_dev_loss(sess, inp, label, dev_dataset, dev_batches, loss)

        epoch += 1

        _logger.info("Epoch {}: train {} dev {}".format(epoch, train_loss, dev_loss))
        _tboard.log_value("loss/train", train_loss / train_batches, step=epoch)
        _tboard.log_value("loss/dev", dev_loss / dev_batches, step=epoch)
        if dev_loss < min(dev_losses):
            save_path = saver.save(sess, os.path.join(weights_path, "{}.ckpt".format(epoch)))
            _logger.info("Saved new best model to {}".format(save_path))
        dev_losses.append(dev_loss)

        if len(dev_losses) > 2:
            losscrit = (dev_losses[-2] - dev_losses[-1]) / dev_losses[-2]
        _tboard.log_value("losscrit", losscrit, step=epoch)
