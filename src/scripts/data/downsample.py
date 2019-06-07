#!/usr/bin/env python3
"""
Downsample an existing dataset.
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

_logger = _util.get_logger(__file__)


def downsample(dataset: str, shape: List[int], partial=False):
    """

    :param dataset:
    :param shape:
    :return:
    """
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(shape, list) and all(isinstance(s, int) for s in shape) and len(shape) == 3
    assert isinstance(partial, bool)
    tf.enable_eager_execution()

    dataset_path = _util.get_rel_datasets_path(dataset)
    _util.ensure_dir(dataset_path)

    data = _dataset.get_dataset(dataset_path, partial=partial)

    resized_dataset = "{}_cropped".format(dataset)
    resized_path = _util.get_rel_datasets_path(resized_dataset)
    _util.ensure_path_free(resized_path)
    _util.mkdir(resized_path)

    iter = data.make_one_shot_iterator()
    records = _dataset.get_records(dataset_path, partial)

    for record in tqdm(records):
        record = record.replace(dataset, resized_dataset)
        sample = iter.next()

        scan = sample[0][0].numpy().squeeze()
        # show_scan(scan, "Original")

        crop = crop_image(scan, 1e-5)
        # show_scan(crop, "Crop")

        factors = [s/d for d, s in zip(crop.shape, shape)]
        resized = ndimage.zoom(crop, zoom=factors, order=4)
        # show_scan(resized, "Resized")

        _dataset.write_record(record, resized, sample[0][1].numpy().squeeze(), sample[1].numpy())

    _dataset.save_shape(resized_path, shape)


def show_scan(img, title):
    plt.figure()
    plt.title(title)
    plt.imshow(img[img.shape[0] // 2, :, :], cmap='Greys')
    # plt.show()
    plt.savefig(_util.get_rel_data_path("imgs", "title.png"))


def crop_image(img, tol=0.):
    # Mask of non-black pixels (assuming image has a single channel).
    mask = img > tol

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    return img[x0:x1, y0:y1, z0:z1]


def main():
    args = _cmd.parse_args_for_callable(downsample)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    downsample(**varsArgs)


if __name__ == '__main__':
    main()
