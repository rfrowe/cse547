"""
Learning and prediction methods for a Support Vector Regressor.
"""
import os

import numpy as np
import tensorboard_logger as _tboard
import tensorflow as tf

import utils.utility as _util
import train.train_test_utils as _tt_utils

_logger = _util.get_logger(__file__)


def learn(sess, encoder, scan, code, features, label, epsilon, train_dataset, train_batches, dev_dataset, dev_batches, lr, weights_path):
    w, b = _get_variables(code)
    saver = tf.train.Saver(var_list=[w, b] + encoder.variables)

    out = _get_svr_output(code, w, b)

    # Define loss function and get ready for training.
    mse = tf.reduce_mean(tf.squared_difference(out, label))
    l2_loss = 0.0001 * tf.nn.l2_loss(w)
    loss = mse + l2_loss

    step = 0
    lr = tf.train.exponential_decay(lr, step, 20, 0.9, staircase=True)
    encoder_op = tf.train.AdamOptimizer(learning_rate=lr * 1e-3).minimize(loss, var_list=encoder.trainable_variables)
    svm_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[w, b])
    train_step = tf.group(encoder_op, svm_op)
    # train_step = svm_op
    sess.run(tf.global_variables_initializer())

    losscrit_count = 0
    losscrit = np.inf
    dev_losses = [np.inf]
    train_iter = train_dataset.shuffle(2 * train_batches).repeat().make_one_shot_iterator().get_next()
    dev_iter = dev_dataset.repeat().make_one_shot_iterator().get_next()
    # sess.graph.finalize()

    while losscrit_count < 10:
        (s, f), y = sess.run(train_iter)
        if y.ndim < 2:
            y = y.reshape((y.shape[-1], 1))

        _, _code, _total_loss, _pred, _mse, l2, bias = \
            sess.run(fetches=[train_step, code, loss, out, mse, l2_loss, b], feed_dict={scan: s, features: f, label: y})
        _tboard.log_value("loss/train", _total_loss, step=step)
        _tboard.log_value("loss/avg_pred", _pred.mean(), step=step)
        _tboard.log_value("params/weights_l2", l2, step=step)
        _tboard.log_value("params/bias", bias, step=step)
        _logger.info("Step {}".format(step + 1))
        _logger.info("Code L2 {}".format(np.linalg.norm(_code, ord=2)))

        step += 1

        if step < 2 or step % 2 == 0:
            dev_loss = _tt_utils.get_dev_loss(sess, scan, features, label, dev_iter, dev_batches, mse) / dev_batches
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
    b = tf.Variable(tf.constant(100.), name="bias")
    return w, b


def predict(weights_path, sess, encoder, scan, features, code, label, dataset, batches):
    w, b = _get_variables(code)

    out = _get_svr_output(code, w, b)

    saver = tf.train.Saver(var_list=[w, b] + encoder.variables)
    saver.restore(sess, weights_path)

    data = dataset.make_one_shot_iterator().get_next()

    _tt_utils.print_error_metrics(sess, scan, features, out, label, data, batches)
