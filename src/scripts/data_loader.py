'''
cd %WS_PATH%\src\scripts data_loader.py --dataset=HCP_1200
'''

import utils.utility as _util, utils.cmd_line as _cmd
import numpy as np, tensorflow as tf, SimpleITK as sitk
import csv, os, pprint
from dltk.io.augmentation import *
from dltk.io.preprocessing import *


def data_loader(dataset: str, batch_size=1, buffer_size=1, iterations=10):
    """
    ------------------------------
    TODO: Docstring
    ------------------------------
    """

    # Parameters
    # TODO: Implement pruning functionality.
    reader_params = {'batch_size': batch_size,
                     'buffer_size': buffer_size,
                     'iterations': iterations}

    # Get image file paths
    # TODO: Correct file paths.  Currently 'all_files' retreives 2 hardcoded images,
    # (used for initial TFR file creation testing), and 'mri_dict' retreives dictionary
    # of images from wrong S3 folder (to demonstrate method of extracting file paths).
    dataset_path = _util.getRelRawPath(dataset)
    all_files = [[dataset_path + '\\100206_SBRef_dc.nii.gz', 100206],
                 [dataset_path + '\\100610_SBRef_dc.nii.gz', 100307]]
    mri_dict = get_mri_dict(dataset)

    # TODO: Also retreive metrics from 'behavioral_summary.csv' for each subject.
    behav_dict = {}

    # Get intelligence scores
    label_dict = get_labels(dataset_path)

    # Create TFRecords file
    tfr_path = dataset_path + '\\train.tfrecords'
    create_TFR_file(tfr_path, all_files)

    # Load tf.data.Dataset from TFRecords file
    dataset = load_TFR_dataset(tfr_path, reader_params)



def get_mri_dict(dataset):

    # Iterate through each subject ID
    mydict = {}
    dataset_path = _util.getRelRawPath(dataset)
    for subj_ID in next(os.walk(dataset_path))[1]:
        subj_path = dataset_path + '\\' + subj_ID

        # Iterate through each MRI resolution
        res_dict = {}
        for res in ['3T', '7T']:
            res_path = subj_path + '\\unprocessed\\' + res
            if (os.path.exists(res_path)):

                # Iterate through MRI categories
                categ_dict = {}
                for categ in next(os.walk(res_path))[1]:
                    categ_path = res_path + '\\' + categ

                    # Get scan names & paths within each category
                    scan_dict = {}
                    for scan in os.listdir(categ_path):
                        if scan.endswith('.gz'):
                            scan_name = scan[10:-7]
                            scan_path = categ_path + '\\' + scan

                            # Append to nested dictionary
                            scan_dict[scan_name] = scan_path
                    categ_dict[categ] = scan_dict
                res_dict[res] = categ_dict
        mydict[subj_ID] = res_dict
    return mydict


def get_labels(dataset_path):
    counter = 0
    label_dict = {}
    with open(dataset_path + '\\behavioral_data.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_dict[row['Subject']] = row['CogFluidComp_Unadj']
    return label_dict


def create_TFR_file(tfr_path, all_files):
    # Open the TFRecords file
    writer = tf.python_io.TFRecordWriter(tfr_path)

    # Write data into a TFRecords file
    for meta_data in all_files:

        # Read the .nii image with SimpleITK and get its numpy array
        sitk_img = sitk.ReadImage(meta_data[0])
        img_arr = sitk.GetArrayFromImage(sitk_img)

        # Take an individual image from the time-series
        # TODO: handle full time series of MRI images for a subject test?
        #img_arr = img_arr[0, :, :, :]

        # Normalize the image to zero mean / unit std dev
        img_arr = whitening(img_arr)

        # Create a tensor with a dummy dimension for channels
        img_arr = img_arr[..., np.newaxis]

        # Assign label
        label = np.int32(meta_data[1])

        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _float_feature(img_arr.ravel())}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    # Close the TFRecords file
    writer.close()


def decode(serialized_example):
    # Decode examples stored in TFRecord (correct image dimensions must be specified)
    features = tf.parse_single_example(
        serialized_example,
        features = {'train/image': tf.FixedLenFeature([72, 104, 90, 1], tf.float32),
                    'train/label': tf.FixedLenFeature([], tf.int64)})

    # Return features as tf.float32 values
    return features['train/image'], features['train/label']

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def load_TFR_dataset(tfr_path, params):
    # Load TFRecords file and decode data
    dataset = tf.data.TFRecordDataset(tfr_path).map(decode)
    dataset = dataset.batch(params['batch_size'])
    dataset = dataset.prefetch(params['buffer_size'])
    return dataset


def main():
    global _logger
    args = _cmd.parseArgsForClassOrScript(data_loader)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)
    #_logger.info("Passed arguments: '{}'".format(varsArgs))

    data_loader(**varsArgs)


if __name__ == '__main__':
    main()

