import os

import numpy as np
import tensorboard_logger as _tboard
import tensorflow as tf
from tqdm import tqdm

import utils.utility as _util
import train.train_utils as _train_utils

_logger = _util.get_logger(__file__)


def learn(sess, inp, code, label, epsilon, train_dataset, train_batches, dev_dataset, dev_batches, lr, weights_path):
    w, b = _get_variables(code)

    saver = tf.train.Saver(var_list=[w, b])

    out = _get_svr_output(code, w, b)

    # Define loss function and get ready for training.
    mse = tf.reduce_mean(tf.squared_difference(out, label))
    l2_loss = 0.1 * tf.nn.l2_loss(w)
    loss = mse + l2_loss

    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[w, b])
    sess.run(tf.global_variables_initializer())


    losscrit_count = 0
    losscrit = np.inf
    dev_losses = [np.inf]
    step = 0
    train_iter = train_dataset.shuffle(train_batches).repeat().make_one_shot_iterator().get_next()
    dev_iter = dev_dataset.repeat().make_one_shot_iterator().get_next()
    # sess.graph.finalize()
    while losscrit_count < 3:
        (x, _), y = sess.run(train_iter)
        if y.ndim < 2:
            y = y.reshape((y.shape[-1], 1))

        _, _total_loss, _pred, _mse, l2, bias = \
            sess.run(fetches=[train_step, loss, out, mse, l2_loss, b], feed_dict={inp: x, label: y})
        _tboard.log_value("loss/train", _total_loss, step=step)
        _tboard.log_value("loss/avg_pred", _pred.mean(), step=step)
        _tboard.log_value("params/weights_l2", l2, step=step)
        _tboard.log_value("params/bias", bias, step=step)
        _logger.info("Step {}".format(step + 1))

        step += 1

        if step < 2 or step % 2 == 0:
            dev_loss = _train_utils.get_dev_loss(sess, inp, label, dev_iter, dev_batches, mse) / dev_batches
            _logger.info("Step {}: dev mse {}".format(step, dev_loss))
            _tboard.log_value("loss/dev", dev_loss, step=step)

            if dev_loss < min(dev_losses):
                save_path = saver.save(sess, os.path.join(weights_path, "{}.ckpt".format(step)))
                _logger.info("Saved new best model to {}".format(save_path))
            dev_losses.append(dev_loss)

            if len(dev_losses) > 2:
                losscrit = (dev_losses[-2] - dev_losses[-1]) / dev_losses[-2]
            _tboard.log_value("loss/crit", losscrit, step=step)

            if losscrit < epsilon:
                losscrit_count += 1
            else:
                losscrit_count = 0


def _get_svr_output(code, w, b):
    return tf.add(tf.matmul(code, w, transpose_b=True), b)


def _get_variables(code):
    w = tf.Variable(tf.random_normal(shape=[1, int(code.shape[-1])]), name="weights")
    b = tf.Variable(tf.constant(80.), name="bias")
    return w, b


def predict(weights_path, sess, inp, code, label, dataset, batches):
    w, b = _get_variables(code)

    out = _get_svr_output(code, w, b)

    # Define loss function and get ready for training.
    loss = tf.reduce_mean(tf.squared_difference(label, out))

    saver = tf.train.Saver(var_list=[w, b])
    saver.restore(sess, weights_path)

    test_iter = dataset.make_one_shot_iterator().get_next()

    total_error = 0
    for batch in tqdm(range(batches)):
        (x, _), y = sess.run(test_iter)
        if y.ndim < 2:
            y = y.reshape((y.shape[-1], 1))


        error, pred = sess.run(fetches=[loss, out], feed_dict={inp: x, label: y})
        # error = (y - pred) ** 2
        _logger.info("Batch {} pred {} truth {} error {}".format(batch + 1, pred, y, error))
        total_error += error

    total_error /= batches
    _logger.info("Average test error: {}".format(total_error))
