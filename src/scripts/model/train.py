#/usr/bin/env python3
import os

import tensorflow as tf
import tensorboard_logger as _tboard
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import utils.cmd_line as _cmd
import utils.utility as _util
from data.hcp_config import SUBJECTS

from model.autoencoder import Autoencoder
from data.dataset import get_dataset, load_shape
from model.tv_loss import total_variation_5d

_logger = _util.get_logger(__file__)


def train(dataset: str, weights: str, epochs=1000, batch_size=64, grad_norm=1000, buffer_size=8, lr=1e-3, l2_reg=1e-1, tv_reg=1e-2, partial=False):
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
    assert isinstance(grad_norm, int) and grad_norm >= 0
    assert isinstance(buffer_size, int) and batch_size > 0
    assert isinstance(lr, float) and lr > 0
    assert isinstance(l2_reg, float) and l2_reg > 0
    assert isinstance(tv_reg, float) and tv_reg > 0
    assert isinstance(partial, bool)

    # Load and ensure required paths.
    weights_path = _get_weights_path(weights)
    log_path = _get_log_path(weights)
    dataset_path = _util.get_rel_datasets_path(dataset)
    _util.ensure_dir(dataset_path)

    # Load model and input shape.
    shape = load_shape(dataset_path)
    model = Autoencoder(l2_reg)

    # Create input/output placeholders.
    inp = tf.placeholder(tf.float32, shape=[None, *shape])
    out = model.call(inp)

    # Initialize loss, reg, and TV
    loss = tf.nn.l2_loss(inp - out)
    if l2_reg > 0:
        loss += tf.add_n(model.losses)
    loss += tf.reduce_sum(total_variation_5d(tf.expand_dims(out, 4)))

    # Configure training operation.
    train_op = _get_train_op(loss, lr, grad_norm)

    # Load datasets
    train_dataset = _get_dataset(os.path.join(dataset_path, "train"), batch_size, buffer_size, partial)
    dev_dataset = _get_dataset(os.path.join(dataset_path, "dev"), batch_size, buffer_size, partial)

    # Setup logging and weight saving.
    _tboard.configure(log_path, flush_secs=5)
    saver = tf.train.Saver()

    # Initialize training loop variables.
    best_dev_loss, dev_loss = np.inf, np.inf
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        _logger.info("Counting datasets...")
        train_batches = _iter_len(sess, train_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tTrain samples: {}".format(train_batches))
        dev_batches = _iter_len(sess, dev_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tDev samples: {}".format(dev_batches))

        for epoch in tqdm(range(epochs)):
            train_iter = train_dataset.make_one_shot_iterator().get_next()

            train_loss = 0
            for _ in range(train_batches):
                _, new_train_loss = sess.run([train_op, loss], feed_dict={inp: sess.run(train_iter)})
                train_loss += new_train_loss

            # Increment before doing anything else to avoid zero-indexed epochs.
            epoch += 1
            _tboard.log_value("epoch", epoch, step=epoch)

            train_loss /= train_batches * batch_size
            _logger.info("Epoch {}: train {}".format(epoch, train_loss))
            _tboard.log_value("train loss", train_loss, step=epoch)

            if epoch % 20 == 0:
                # Compute and log dev loss
                new_dev_loss = _get_dev_loss(sess, inp, dev_dataset, dev_batches, batch_size, loss)
                _logger.info("Epoch {}: dev {} diff {}".format(epoch, new_dev_loss, dev_loss - new_dev_loss))
                dev_loss = new_dev_loss
                if dev_loss < best_dev_loss:
                    save_path = saver.save(sess, os.path.join(weights_path, "{}.ckpt".format(epoch)))
                    _logger.info("Saved new best model to {}".format(save_path))
                    best_dev_loss = new_dev_loss

                # Plot some reconstruction images
                _log_reconstruction_imgs("eval", sess, dev_dataset, inp, out, epoch, weights_path)
                _log_reconstruction_imgs("train", sess, train_dataset, inp, out, epoch, weights_path)


def _log_reconstruction_imgs(title, sess, dataset, inp, out, epoch, weights_path):
    data = dataset.make_one_shot_iterator().get_next()

    images = []
    scans = sess.run(data)
    recons = sess.run(out, feed_dict={inp: scans})
    for i, (scan, recon) in enumerate(zip(scans, recons)):
        scan_cross = scan[:, scan.shape[1] // 2, :]
        recon_cross = recon[:, scan.shape[1] // 2, :]

        plt.figure()
        plt.title(title)
        _, axarr = plt.subplots(ncols=2)
        axarr[0].title.set_text("Original")
        axarr[0].imshow(scan_cross)
        axarr[1].title.set_text("Reconstructed")
        axarr[1].imshow(recon_cross)
        plt.savefig(os.path.join(weights_path, "epoch{}_{}.png".format(epoch, i)))

        combined = np.hstack((scan_cross, recon_cross))
        images.append(combined)
    _tboard.log_images("{} reconstruction".format(title), images, step=epoch)


def _get_train_op(loss, lr, grad_norm):
    op = tf.train.AdamOptimizer(learning_rate=lr)
    if grad_norm > 0:
        tvars = tf.trainable_variables()
        grads = tf.gradients(loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=grad_norm)
        grads_and_vars = zip(grads, tvars)
    else:
        grads_and_vars = op.compute_gradients(loss)
    train_op = op.apply_gradients(grads_and_vars)
    return train_op


def _get_weights_path(weights: str) -> str:
    weights_path = _util.get_rel_weights_path(weights)
    _util.ensure_path_free(weights_path, empty_ok=True)
    _util.mkdir(weights_path)
    return weights_path


def _get_log_path(weights: str) -> str:
    log_path = _util.get_rel_log_path(weights)
    _util.ensure_path_free(log_path, empty_ok=True)
    _util.mkdir(log_path)
    return log_path


def _visualize_recons(sess, inp, sample, out, weights_path, filename):
    scan = sess.run(sample)
    recon = sess.run(out, feed_dict={inp: scan}).squeeze()
    scan = scan.squeeze()

    _, axarr = plt.subplots(ncols=2)
    axarr[0].imshow(scan[scan.shape[0] // 2, :, :])
    axarr[1].imshow(recon[recon.shape[0] // 2, :, :])
    plt.savefig(os.path.join(weights_path, "{}.png".format(filename)))


def _iter_len(sess, data):
    count = 0
    for _ in tqdm(SUBJECTS):
        try:
            sess.run(data)
            count += 1
        except tf.errors.OutOfRangeError:
            return count
    return count


def _get_dev_loss(sess, inp, data, num_batches, batch_size, loss):
    data = data.make_one_shot_iterator().get_next()
    data_len = num_batches * batch_size

    _logger.info("Calculating dev loss...")

    value = 0.
    for _ in tqdm(range(num_batches)):
        value += sess.run(fetches=loss, feed_dict={inp: sess.run(data)}) / data_len

    return value


def _get_dataset(name, batch_size, buffer_size, partial):
    dataset_path = _util.get_rel_datasets_path(name)
    _util.ensure_dir(dataset_path)

    return get_dataset(dataset_path, batch_size=batch_size, buffer_size=buffer_size, partial=partial).map(_only_cropped_scan)


def _only_cropped_scan(*data):
    return data[0][0]


def main():
    args = _cmd.parse_args_for_callable(train)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    train(**varsArgs)


if __name__ == '__main__':
    main()

