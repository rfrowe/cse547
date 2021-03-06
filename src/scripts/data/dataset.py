#!/usr/bin/env python3
"""
Tool for generating preprocessed TFRecord dataset.
"""

import csv
import functools
import glob
import json
import os

import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
from typing import List

import utils.cmd_line as _cmd


from data.hcp_config import FEATURES, LABEL, SUBJECTS, SHAPE
from utils import utility as _util

_logger = _util.get_logger(__file__)


def generate(raw: str, dataset=None, scan_dir="T1w/T1w_acpc_dc_restore_brain.nii.gz", overwrite=False, partial=False, validate_scans=True):
    """
    Generates a TFRecords dataset for HCP in the given raw directory.

    :param raw: Directory containing data in the HCP1200 format. Must contain behavioral_data.csv.
    :param dataset: Directory in which to put the resultant dataset, or None for same as raw.
    :param scan_dir: Location of MRI scan inside each participant to use.
    :param overwrite: If dataset already exists, overwrite it. Otherwise, will fail.
    :param partial: If any subject's data are not found, continue anyway. Otherwise, will fail.
    :param validate_scans: Verify before beginning that all necessary scans exist. This may take a while.
    """
    assert isinstance(raw, str) and len(raw)
    assert dataset is None or isinstance(dataset, str) and len(dataset)
    assert isinstance(scan_dir, str) and len(scan_dir)
    assert isinstance(overwrite, bool)
    assert isinstance(partial, bool)
    assert isinstance(validate_scans, bool)

    if dataset is None:
        dataset = raw

    raw_path = _util.get_rel_raw_path(raw)
    _util.ensure_dir(raw_path)

    behavioral = _get_behavioral_data(_util.get_rel_raw_path(), partial)
    dataset_path = _get_dataset_path(dataset, overwrite)

    # Ensure scans exist for all subjects before beginning
    if validate_scans:
        for subject in tqdm(SUBJECTS):
            scan_path = os.path.join(raw_path, subject, scan_dir)
            if not os.path.exists(scan_path) and partial:
                continue
            _util.ensure_file(scan_path)

    for i, subject in tqdm(enumerate(sorted(SUBJECTS)), total=len(SUBJECTS)):
        assert isinstance(subject, str) and len(subject)
        _logger.info("Processing subject {} ({}/{})".format(subject, i + 1, len(SUBJECTS)))

        subject_path = os.path.join(raw_path, subject)
        if not os.path.exists(subject_path) and partial:
            continue
        if not validate_scans:
            _util.ensure_dir(subject_path)

        record_path = _get_record_path(dataset_path, subject, overwrite)

        scan_path = os.path.join(subject_path, scan_dir)
        _util.ensure_file(scan_path)

        scan = sitk.ReadImage(scan_path)
        arr = sitk.GetArrayFromImage(scan)
        assert arr.ndim == 3, "Expected 3-d im scans, got {}".format(arr.ndim)

        assert subject in behavioral, "Subject not found in behavioral data: {}".format(subject)
        vector, label = behavioral[subject]

        write_record(record_path, arr.ravel(), vector, label)
    save_shape(dataset_path, SHAPE)


def write_record(record_path: str, scan, behavioral, label):
    features = {
        "scan": tf.train.Feature(float_list=tf.train.FloatList(value=scan.ravel())),
        "behavioral": tf.train.Feature(float_list=tf.train.FloatList(value=behavioral)),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label]))
    }

    record = tf.train.Example(features=tf.train.Features(feature=features))
    with tf.python_io.TFRecordWriter(record_path) as writer:
        writer.write(record.SerializeToString())


def _get_dataset_path(dataset: str, overwrite: bool):
    path = _util.get_rel_datasets_path(dataset)
    if os.path.exists(path):
        if overwrite:
            _util.rm(path)
        else:
            _util.ensure_path_free(path)

    _logger.info("Creating {}".format(path))
    _util.mkdir(path)
    return path


def _get_record_path(dataset_path: str, subject: str, overwrite: bool):
    path = os.path.join(dataset_path, "{}.tfrecords".format(subject))
    if os.path.exists(path):
        if overwrite:
            _util.rm(path)
        else:
            _util.ensure_path_free(path)
    return path


def _get_feature(value):
    if value == "TRUE" or value == "M":
        return 1.
    elif len(value) == 0 or value == "FALSE" or value == "F":
        return 0.
    elif value[0].isnumeric() and ("-" in value or "+" in value):
        if "-" in value:
            return float(value[:value.index("-")])
        else:
            return float(value[:value.index("+")])
    else:
        return float(value)


def _get_behavioral_data(raw_path: str, partial: bool):
    behavioral_path = os.path.join(raw_path, "behavioral_data.csv")
    _util.ensure_file(behavioral_path)

    subjects = set(SUBJECTS)
    data = {}

    with open(behavioral_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subject = row["Subject"]

            if subject in subjects:
                features = []
                for feature in FEATURES:
                    features.append(_get_feature(row[feature]))

                data[subject] = (features, float(row[LABEL]))

    for subject in SUBJECTS:
        assert subject in subjects or partial, \
            "Missing subject from behavioral data: {}".format(subject)
    return data


def _decode(shape, serialized_example):
    # Decode examples stored in TFRecord (correct image dimensions must be specified)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "scan": tf.FixedLenFeature(shape, tf.float32),
            "behavioral": tf.FixedLenFeature([len(FEATURES)], tf.float32),
            "label": tf.FixedLenFeature([], tf.float32)
        }
    )

    return (features["scan"], features["behavioral"]), features["label"]


def get_records(dataset_path: str, partial: bool):
    assert isinstance(dataset_path, str) and len(dataset_path)
    assert isinstance(partial, bool)
    _util.ensure_dir(dataset_path)

    records = sorted(glob.glob(os.path.join(dataset_path, "*.tfrecords")))
    saved_subjects = set(os.path.splitext(os.path.basename(record))[0] for record in records)
    subjects = set(SUBJECTS)

    missing = subjects - saved_subjects
    extra = saved_subjects - subjects

    assert len(missing) == 0 or partial, "Missing records: {}".format(missing)
    assert len(extra) == 0, "Extra records: {}".format(extra)
    return records


def load_shape(dataset_path: str) -> List[int]:
    if dataset_path.endswith("train") or dataset_path.endswith("dev") or dataset_path.endswith("test"):
        dataset_path = os.path.join(dataset_path, os.pardir)
    with open(os.path.join(dataset_path, "shape.json"), "r") as f:
        shape = json.load(f)
    assert isinstance(shape, list) and all(isinstance(s, int) and s > 0 for s in shape) and len(shape) == 3, \
        "Invalid shape: {}".format(shape)
    return shape


def save_shape(dataset_path: str, shape: List[int]):
    assert isinstance(shape, list) and all(isinstance(s, int) and s > 0 for s in shape) and len(shape) == 3, \
        "Invalid shape: {}".format(shape)
    with open(os.path.join(dataset_path, "shape.json"), "w") as f:
        json.dump(shape, f)


def load_mean(dataset_path: str):
    return np.load(os.path.join(dataset_path, "mean.pickle"), allow_pickle=True)


def save_mean(dataset_path: str, mean: np.ndarray):
    mean.dump(os.path.join(dataset_path, "mean.pickle"))


def load_std(dataset_path: str):
    return np.load(os.path.join(dataset_path, "std.pickle"), allow_pickle=True)


def save_std(dataset_path: str, mean: np.ndarray):
    mean.dump(os.path.join(dataset_path, "std.pickle"))


def get_dataset(dataset_path: str, partial=False):
    records = get_records(dataset_path, partial)

    shape = load_shape(dataset_path)

    return tf.data.TFRecordDataset(records).map(functools.partial(_decode, shape))


def get_dataset_by_name(name, partial):
    dataset_path = _util.get_rel_datasets_path(name)
    _util.ensure_dir(dataset_path)

    return get_dataset(dataset_path, partial=partial)


def main():
    args = _cmd.parse_args_for_callable(generate)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    _logger.info("Passed arguments: '{}'".format(varsArgs))

    generate(**varsArgs)


if __name__ == '__main__':
    main()
