#!/usr/bin/env python3
"""
standardize an existing dataset.
"""
from typing import List

from scipy import ndimage
from tqdm import tqdm

import utils.utility as _util
import utils.cmd_line as _cmd

import data.dataset as _dataset

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

from data.downsample import show_scan

_logger = _util.get_logger(__file__)


def standardize(dataset: str):
    """

    :param dataset:
    :return:
    """
    assert isinstance(dataset, str) and len(dataset)

    tf.enable_eager_execution()

    train_path = _util.get_rel_datasets_path(dataset, "train")
    _util.ensure_dir(train_path)

    dataset_path = _util.get_rel_datasets_path(dataset)

    standardized_name = _get_standardized_name(dataset)
    standardized_path = _util.get_rel_datasets_path(standardized_name)
    # _util.ensure_path_free(standardized_path, empty_ok=True)
    # _util.mkdir(standardized_path)

    train_data = _dataset.get_dataset(train_path, partial=True)
    train_iter = train_data.repeat().make_one_shot_iterator()
    train_records = _dataset.get_records(train_path, partial=True)

    # Compute sample mean over train
    total = train_iter.next()[0][0]
    for _ in tqdm(train_records[1:]):
        sample = train_iter.next()

        total += sample[0][0]
    mean = total / len(train_records)

    total = tf.square(train_iter.next()[0][0] - mean)
    for _ in tqdm(train_records[1:]):
        sample = train_iter.next()

        scan = sample[0][0]
        total += tf.square(scan - mean)
    std = tf.sqrt(tf.reduce_mean(total))

    _standardize_dataset(train_path, dataset, mean, std)
    _standardize_dataset(_util.get_rel_datasets_path(dataset, "dev"), dataset, mean, std)
    _standardize_dataset(_util.get_rel_datasets_path(dataset, "test"), dataset, mean, std)

    _dataset.save_shape(standardized_path, _dataset.load_shape(dataset_path))
    _dataset.save_mean(standardized_path, mean.numpy())
    _dataset.save_std(standardized_path, std.numpy())


def _standardize_dataset(dataset_path, dataset, mean, std):
    data = _dataset.get_dataset(dataset_path, partial=True).make_one_shot_iterator()
    records = _dataset.get_records(dataset_path, partial=True)

    standardized_name = _get_standardized_name(dataset)
    standardized_path = dataset_path.replace(dataset, standardized_name)
    _util.ensure_path_free(standardized_path, empty_ok=True)
    _util.mkdir(standardized_path)
    for record in tqdm(records):
        record = record.replace(dataset, standardized_name)
        sample = data.next()

        scan = sample[0][0]
        # show_scan(scan.numpy().squeeze(), "Original")

        standardized = (scan - mean) / std
        # show_scan(standardized.numpy().squeeze(), "Standardized")

        _dataset.write_record(record, standardized.numpy().squeeze(), sample[0][1].numpy().squeeze(), sample[1].numpy())


def _get_standardized_name(dataset):
    standardized_name = "{}_standardized".format(dataset)
    return standardized_name


def main():
    args = _cmd.parse_args_for_callable(standardize)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    standardize(**varsArgs)


if __name__ == '__main__':
    main()
