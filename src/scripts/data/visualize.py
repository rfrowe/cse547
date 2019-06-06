#!/usr/bin/env python3
"""
Visualize an existing dataset.
"""
import utils.utility as _util
import utils.cmd_line as _cmd

import data.dataset as _dataset

import tensorflow as tf

from data.downsample import show_scan

_logger = _util.get_logger(__file__)


def visualize(dataset: str):
    """

    :param dataset:
    :return:
    """
    assert isinstance(dataset, str) and len(dataset)
    tf.enable_eager_execution()

    dataset_path = _util.get_rel_datasets_path(dataset)
    _util.ensure_dir(dataset_path)

    data = _dataset.get_dataset(dataset_path, 1, 1, partial=True)

    scan = data.make_one_shot_iterator().next()[0][0].numpy()
    show_scan(scan.squeeze(), "")


def main():
    args = _cmd.parse_args_for_callable(visualize)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    visualize(**varsArgs)


if __name__ == '__main__':
    main()
