'''
cd %WS_PATH%\src\scripts data_loader.py --dataset=HCP_1200
'''

import utils.utility as _util, utils.cmd_line as _cmd
import numpy as np, tensorflow as tf, SimpleITK as sitk
import csv, os, pprint
#from dltk.io.augmentation import *
#from dltk.io.preprocessing import *


def data_loader(dataset: str, batch_size=1, buffer_size=1, local_test=0):
    """
    ------------------------------------------------------------------------------------------
    Dataloader:
        - Saves a .tfrecords file with each subject's data in \\data\\raw\\HCP_1200\\TFRecords.
        - Metrics are loaded according to codes in \\data\\raw\\HCP_1200\\behavioral_data_pruned.csv:
            - Marked with '0': not included as features
            - Marked with '1': included as features
            - Marked with '2': included as label
        - Set optional param '--local_test=1' to test with dummy data on local machine.
    
        Returns:
        1) Dataset of type 'tf.data.TFRecordDataset' containing MRI image file paths and
            behavioral metrics as dictionaries with 6-digit subject_ids as keys.
        2) Dictionary containing intel scores with 6-digit subject_ids as keys.

    TODO: Expand docstring
    ------------------------------------------------------------------------------------------
    """

    # Parameters
    reader_params = {'batch_size': batch_size,
                     'buffer_size': buffer_size}
    dataset_path = _util.getRelRawPath(dataset)
    tfr_path = dataset_path + '\\TFRecords'

    # Get image file paths, behav metrics, & intel scores as dicts with subjects as keys
    mri_dict = get_mri_dict(dataset_path)
    behav_dict = get_behav_dict(dataset_path)
    intel_dict = get_intel_dict(dataset_path)

    # Create TFRecords file
    create_TFR_file(tfr_path, mri_dict, behav_dict, intel_dict, local_test)

    # Load tf.data.Dataset from TFRecords file
    dataset = load_TFR_dataset(tfr_path, reader_params)

    return dataset, intel_dict


def get_mri_dict(dataset_path):
    # Main MRI dictionary of all subjects
    mri_dict = {}
    
    # Iterate through each subject ID
    for subj_ID in next(os.walk(dataset_path))[1]:
        subj_dict = {}
        subj_path = dataset_path + '\\' + subj_ID

        # Add scans in 'Diffusion' folder to subject dictionary
        diff_path = subj_path + '\\T1w\\Diffusion'
        if os.path.exists(diff_path):
            for file_name in os.listdir(diff_path):
                if file_name.endswith('.gz'):
                    img_name = 'Diffusion_' + file_name[:-7]
                    img_path = diff_path + '\\' + file_name
                    subj_dict[img_name] = img_path

        # Add scans in 'Diffusion_7T' folder to subject dictionary
        diff_7T_path = subj_path + '\\T1w\\Diffusion_7T'
        if os.path.exists(diff_7T_path):
            for file_name in os.listdir(diff_7T_path):
                if file_name.endswith('.gz'):
                    img_name = 'Diffusion_7T_' + file_name[:-7]
                    img_path = diff_7T_path + '\\' + file_name
                    subj_dict[img_name] = img_path

        # Add scans in 'Results' folder to subject dictionary
        results_path = subj_path + '\\T1w\\Results'
        if os.path.exists(results_path):
            for folder_name in next(os.walk(results_path))[1]:
                for file_name in os.listdir(results_path + '\\' + folder_name):
                    if file_name.endswith('.gz'):
                        img_name = folder_name + file_name[:-7]
                        img_path = results_path + '\\' + folder_name + '\\' + file_name
                        subj_dict[img_name] = img_path

        # Add individual subject dictionary to main MRI dictionary
        mri_dict[subj_ID] = subj_dict

    # Return main MRI dictionary of all subjects
    return mri_dict


def get_behav_dict(dataset_path):
    # Get list of pruned metrics
    metrics = []
    for line in open(dataset_path + '\\behavioral_data_pruned.csv').read().splitlines():
        if line.split(',')[2] == '1':
            metrics.append(line.split(',')[0])

    # Create dictionary of metrics for each subject
    # TODO: Handle missing subject values.
    behav_dict = {}
    with open(dataset_path + '\\behavioral_data.csv') as f:
        for row in csv.DictReader(f):
            single_subj_behav_dict = {}
            for m in metrics:
                single_subj_behav_dict[m] = row[m]
            behav_dict[row['Subject']] = single_subj_behav_dict
    
    return behav_dict


def get_intel_dict(dataset_path):
    # Get intelligence metric
    # TODO: Possibility of multiple intelligence metrics?
    for line in open(dataset_path + '\\behavioral_data_pruned.csv').read().splitlines():
        if line.split(',')[2] == '2':
            metric = line.split(',')[0]
    
    # Create dictionary of intelligence scores for each subject
    with open(dataset_path + '\\behavioral_data.csv') as f:
        intel_dict = {}
        for row in csv.DictReader(f):
            score = row[metric]
            # TODO: Properly handle missing subject values
            if score == "":
                score = 100
            intel_dict[row['Subject']] = float(score)

    return intel_dict


def create_TFR_file(tfr_path, mri_dict, behav_dict, intel_dict, local_test):
    # Iterate through each subject ID
    if local_test: subj_counter = 0
    for subj_id in mri_dict:
        if local_test: subj_counter += 1
        if local_test and subj_counter > 2: break
    
        # Open the TFRecords file (x_feature)
        writer = tf.python_io.TFRecordWriter(tfr_path + '\\' + subj_id + '_x.tfrecords')
        x_feature_dict = {}

        # Add MRI images to x_feature dictionary
        if local_test: mri_counter = 0
        for mri in mri_dict[subj_id]:
            if local_test: mri_counter += 1
            if local_test and mri_counter > 3: break

            # Reaad the .nii.gz image with SimpleITK and get its numpy array
            sitk_img = sitk.ReadImage(mri_dict[subj_id][mri])
            img_arr = sitk.GetArrayFromImage(sitk_img)

            # If image is a 4D time-series, take only an individual image
            # TODO: How to handle 4D time-series?  Currently just taking first image.
            if len(img_arr.shape) == 4:
                img_arr = img_arr[0, :, :, :]

            # Create a tensor with a dummy dimension for channels
            img_arr = img_arr[..., np.newaxis]

            # Add to dictionary
            x_feature_dict[mri] = _float_arr_feature(img_arr.ravel())

        # Add behavioral metrics to x_feature dictionary
        for behav in behav_dict[subj_id]:
            x_feature_dict[behav] = _str_feature(behav_dict[subj_id][behav])

        # Create the feature
        features = tf.train.Features(feature=x_feature_dict)

        # Create an example protocol buffer
        example = tf.train.Example(features=features)

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # Close the TFRecords file
        writer.close()


        # TODO: Add y_features into TFRecords file.
        '''
        # Open the TFRecords file (y_feature)
        writer = tf.python_io.TFRecordWriter(tfr_path + '\\' + subj_id + '_y.tfrecords')

        # Assign label
        y_feature = _float_feature(intel_dict[subj_id])
        features = tf.train.Features(feature=y_feature)

        # Create an example protocol buffer
        example = tf.train.Example(features=features)

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # Close the TFRecords file
        writer.close()
        '''


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

def _float_arr_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _str_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


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

