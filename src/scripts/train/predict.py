#!/usr/bin/env python3
"""
Predict fluid intelligence over test set given a trained regression model.

"""
import os

import tensorflow as tf

import data.dataset as _dataset
import model.svr as _svr
import utils.cmd_line as _cmd
import utils.utility as _util

import train.train_test_utils as _tt_utils
import data.hcp_config as _hcp

_logger = _util.get_logger(__file__)


def predict(dataset: str, encoder_weights: str, model: str, model_weights: str):
    """
    Assess regression model performance

    :param dataset: Name of dataset over which to test.
    :param encoder_weights: Path to trained encoder weights.
    :param model: Model type to use for regression
    :param model_weights: Path to trained regression weights.
    """
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(encoder_weights, str) and len(encoder_weights)
    assert isinstance(model, str) and len(model)
    assert isinstance(model_weights, str) and len(model_weights)

    model = _get_model(model)

    if not os.path.isabs(encoder_weights):
        encoder_weights = _util.get_rel_weights_path(encoder_weights)
        _util.ensure_dir(os.path.dirname(encoder_weights))
    if not os.path.isabs(model_weights):
        model_weights = _util.get_rel_weights_path(model_weights)
        _util.ensure_dir(os.path.dirname(model_weights))

    test_dataset = _dataset.get_dataset_by_name(os.path.join(dataset, "test"), partial=True).batch(1)
    shape = _dataset.load_shape(_util.get_rel_datasets_path(dataset))

    label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    features = tf.placeholder(dtype=tf.float32, shape=[None, len(_hcp.FEATURES)])

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        # Define input, output, and intermediate operation.
        encoder, (scan, code) = _tt_utils.load_encoder(sess, encoder_weights, 1, shape)

        _logger.info("Counting dataset...")
        test_batches = _tt_utils.dataset_iter_len(sess, test_dataset.make_one_shot_iterator().get_next())
        _logger.info("\tTest samples: {}".format(test_batches))

        model(model_weights, sess, encoder, scan, features, code, label, test_dataset, test_batches)


def _get_model(model: str) -> callable:
    model = model.lower()
    if model == "svr":
        return _svr.predict
    else:
        # TODO: add more regression models.
        raise NotImplementedError("Model '{}' not implemented".format(model))


def main():
    args = _cmd.parse_args_for_callable(predict)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)

    predict(**varsArgs)


if __name__ == "__main__":
    main()
