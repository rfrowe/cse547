#/usr/bin/env python3
import os
from collections import defaultdict

import tensorflow as tf
import tensorboard_logger as _tboard
import numpy as np

from tqdm import tqdm

import utils.cmd_line as _cmd
import utils.utility as _util
from data.hcp_config import SUBJECTS

from model.autoencoder import Autoencoder
import data.dataset as _dataset
from model.tv_loss import total_variation_5d

_logger = _util.get_logger(__file__)


def train(dataset: str, epochs: int, batch_size: int, buffer_size: int, lr: float, l2_reg=0., tv_reg=0., ssim_loss=0., sobel_loss=0.):
    """
    Trains an Autoencoder using the specified parameters.

    :param dataset: Existing dataset over which to train. Must contain train, dev, {mean,std}.pickle, shape.json
    :param epochs: Number of iterations over training data before termination.
    :param batch_size: Number of training samples per batch.
    :param buffer_size: Number of batches to prefetch.
    :param lr: Adam optimization initial learning rate.
    :param l2_reg: L2 regularization coefficient for kernel weights.
    :param tv_reg: Total Variation regularization coefficient for data.
    :param ssim_loss: SSIM regularization coefficient for data.
    :param sobel_loss: L2 regularization coefficient for data Sobel difference.
    :return:
    """
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(epochs, int) and epochs > 0
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(buffer_size, int) and batch_size > 0
    assert isinstance(lr, float) and lr > 0
    assert isinstance(l2_reg, float) and l2_reg >= 0
    assert isinstance(tv_reg, float) and tv_reg >= 0
    assert isinstance(ssim_loss, float) and ssim_loss >= 0
    assert isinstance(sobel_loss, float) and sobel_loss >= 0

    # Load and ensure required paths.
    weights_path = _util.get_weights_path_by_param(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        l2_reg=l2_reg,
        tv_reg=tv_reg,
        ssim_loss=ssim_loss,
        sobel_loss=sobel_loss
    )
    log_path = os.path.join(weights_path, "logs")
    _util.ensure_path_free(log_path, empty_ok=True)
    _util.mkdir(log_path)
    dataset_path = _util.get_rel_datasets_path(dataset)
    _util.ensure_dir(dataset_path)

    # Load model and input shape.
    shape = _dataset.load_shape(dataset_path)
    mean = _dataset.load_mean(dataset_path)
    std = _dataset.load_std(dataset_path)
    model = Autoencoder(l2_reg)

    # Create input/output placeholders.
    inp = tf.image.per_image_standardization(tf.placeholder(tf.float32, shape=[None, *shape]))
    out = model.call(inp)

    # Initialize loss functions.
    total_loss, l2_loss, l2_reg, tv_reg, ssim_loss, sobel_loss = \
        _get_losses(inp, out, batch_size, model.losses, l2_reg, tv_reg, ssim_loss, sobel_loss)
    # Configure training operation.
    train_op = _get_train_op(total_loss, lr)

    # Load datasets
    train_dataset = (_dataset
                     .get_dataset(os.path.join(dataset_path, "train"), partial=True)
                     .map(_only_cropped_scan)
                     .batch(batch_size)
                     .prefetch(buffer_size))
    dev_dataset = (_dataset
                   .get_dataset(os.path.join(dataset_path, "dev"), partial=True)
                   .map(_only_cropped_scan)
                   .batch(batch_size)
                   .prefetch(buffer_size))

    # Setup logging and weight saving.
    _tboard.configure(log_path, flush_secs=2)
    saver = tf.train.Saver()

    # Initialize training loop variables.
    best_dev_loss, dev_loss = np.inf, np.inf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        _logger.info("Counting datasets...")
        train_batches = _iter_len(sess, train_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tTrain samples: {}".format(train_batches))
        dev_batches = _iter_len(sess, dev_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tDev samples: {}".format(dev_batches))

        train_loss = total_loss / train_batches
        dev_loss = total_loss / dev_batches

        train_dataset = (_dataset.get_dataset(os.path.join(dataset_path, "train"), partial=True)
                         .map(_only_cropped_scan)
                         .batch(batch_size)
                         .prefetch(buffer_size))

        for epoch in tqdm(range(epochs)):
            train_iter = train_dataset.make_one_shot_iterator().get_next()

            losses = defaultdict(float)
            for _ in range(train_batches):
                sample = sess.run(train_iter)
                _, _train_loss, _l2_loss, _l2_reg, _tv_reg, _ssim_loss, _sobel_loss = \
                    sess.run(
                        [train_op, train_loss, l2_loss, l2_reg, tv_reg, ssim_loss, sobel_loss],
                        feed_dict={inp: sample})
                losses["train/loss/total"] += _train_loss
                losses["train/loss/l2_loss"] += _l2_loss
                losses["train/reg/l2"] += _l2_reg
                losses["train/reg/tv"] += _tv_reg
                losses["train/loss/ssim"] += _ssim_loss
                losses["train/loss/sobel"] += _sobel_loss

            # Increment before doing anything else to avoid zero-indexed epochs.
            epoch += 1

            # Log training losses to tensorboard.
            for name, val in losses.items():
                _tboard.log_value(name, val, step=epoch)
            _logger.info("Epoch {}: train loss {}".format(epoch, losses["train/loss/total"]))

            # Compute dev metrics every 2 epochs.
            if epoch < 2 or epoch % 2 == 0:
                losses.clear()

                # Compute and log dev loss
                _dev_loss, _l2_loss, _l2_reg, _tv_reg, _ssim_loss, _sobel_loss = \
                    _get_dev_loss(sess, inp, dev_dataset, dev_batches, dev_loss, l2_loss, l2_reg, tv_reg, ssim_loss, sobel_loss)

                # Log dev losses to tensorboard.
                _logger.info("Epoch {}: dev loss {}".format(epoch, _dev_loss))

                _tboard.log_value("dev/loss/total", _dev_loss, step=epoch)
                _tboard.log_value("dev/loss/l2_loss", _l2_loss, step=epoch)
                _tboard.log_value("dev/reg/l2", _l2_reg, step=epoch)
                _tboard.log_value("dev/reg/tv", _tv_reg, step=epoch)
                _tboard.log_value("dev/loss/ssim", _ssim_loss, step=epoch)
                _tboard.log_value("dev/loss/sobel", _sobel_loss, step=epoch)

                # Save best model.
                if _dev_loss < best_dev_loss:
                    save_path = saver.save(sess, os.path.join(weights_path, "{}.ckpt".format(epoch)))
                    _logger.info("Saved new best model to {}".format(save_path))
                    best_dev_loss = _dev_loss

                # Plot some reconstruction images
                _logger.info("Generating reconstruction plots...")
                _log_reconstruction_imgs("eval", sess, train_dataset, inp, out, epoch, mean, std)
                _log_reconstruction_imgs("train", sess, train_dataset, inp, out, epoch, mean, std)


def _log_reconstruction_imgs(title, sess, dataset, inp, out, epoch, mean, std):
    data = dataset.make_one_shot_iterator().get_next()

    images = []
    standardized = []
    scans = sess.run(data)
    recons = sess.run(out, feed_dict={inp: scans})
    for i, (scan, recon) in enumerate(zip(scans, recons)):
        scan_cross = np.array(scan[scan.shape[1] // 2, :, :])
        recon_cross = np.array(recon[scan.shape[1] // 2, :, :])
        standardized.append(
            np.hstack(
                (_normalize_image(scan_cross),
                 _normalize_image(recon_cross)))
        )

        scan *= std
        scan += mean
        recon *= std
        recon += mean
        scan_cross = np.array(scan[scan.shape[1] // 2, :, :])
        recon_cross = np.array(recon[scan.shape[1] // 2, :, :])
        images.append(
            np.hstack(
                (_normalize_image(scan_cross),
                 _normalize_image(recon_cross)))
        )

        # plt.figure()
        # plt.title(title)
        # _, axarr = plt.subplots(ncols=2)
        # axarr[0].title.set_text("Original")
        # axarr[0].imshow(scan_cross)
        # axarr[1].title.set_text("Reconstructed")
        # axarr[1].imshow(recon_cross)
        # plt.savefig(os.path.join(weights_path, "epoch{}_{}.png".format(epoch, i)))
        # plt.close('all')

    _tboard.log_images("{} reconstruction".format(title), images, step=epoch)
    _tboard.log_images("{} reconstruction (standardized)".format(title,), standardized, step=epoch)


def _normalize_image(img):
    img -= img.min()
    img /= img.max()
    img *= 256
    return img


def _get_train_op(loss, lr):
    return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # if grad_norm > 0:
    #     tvars = tf.trainable_variables()
    #     grads = tf.gradients(loss, tvars)
    #     grads, _ = tf.clip_by_global_norm(grads, clip_norm=grad_norm)
    #     grads_and_vars = zip(grads, tvars)
    # else:
    #     grads_and_vars = op.compute_gradients(loss)
    # train_op = op.apply_gradients(grads_and_vars)


def _get_losses(inp, out, batch_size, reg_losses, l2_reg, tv_reg, ssim_loss, sobel_loss):
    l2_loss = tf.nn.l2_loss(inp - out) / batch_size
    # l2_loss = tf.reduce_mean(tf.squared_difference(inp, out))
    total_loss = l2_loss

    if l2_reg > 0:
        l2_reg = l2_reg * tf.add_n(reg_losses)
        total_loss += l2_reg
    else:
        l2_reg = tf.Variable(initial_value=0, trainable=False)

    if tv_reg > 0:
        tv_reg = tv_reg * tf.reduce_sum(total_variation_5d(tf.expand_dims(out, 4))) / batch_size
        total_loss += tv_reg
    else:
        tv_reg = tf.Variable(initial_value=0, trainable=False)

    if ssim_loss > 0:
        ssim_loss = ssim_loss * tf.reduce_mean(tf.image.ssim(inp, out, 1.)) / batch_size
        total_loss += ssim_loss
    else:
        ssim_loss = tf.Variable(initial_value=0, trainable=False)

    if sobel_loss > 0:
        sobel_loss = sobel_loss * tf.reduce_sum(tf.squared_difference(tf.image.sobel_edges(inp), tf.image.sobel_edges(out))) / batch_size
        total_loss += sobel_loss
    else:
        sobel_loss = tf.Variable(initial_value=0, trainable=False)
    return total_loss, l2_loss, l2_reg, tv_reg, ssim_loss, sobel_loss


def _iter_len(sess, data):
    count = 0
    for _ in tqdm(SUBJECTS):
        try:
            sess.run(data)
            count += 1
        except tf.errors.OutOfRangeError:
            return count
    return count


def _get_dev_loss(sess, inp, data, num_batches, *losses):
    data = data.make_one_shot_iterator().get_next()

    _logger.info("Calculating dev losses...")

    loss_vals = np.zeros(len(losses))
    for _ in tqdm(range(num_batches)):
        loss_vals += np.array(sess.run(fetches=losses, feed_dict={inp: sess.run(data)}))

    return loss_vals


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

