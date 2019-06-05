# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:57:22 2019

@author: Lynda



"""

import numpy as np
import tensorflow as tf
import os

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tqdm import tqdm

import utils.cmd_line as _cmd
import utils.utility as _util

from data.dataset import get_dataset, load_shape
from model.train import _get_dataset, _only_cropped_scan

#TODOS
# Kernel Functions if wanted, 
# formatting & structure for compatibility with network output
# main and/or supporting functions?
# find relative path to tfrecords


# from hcp_config, get subjids and behavioral data attributes

# split subjids into train and tes

# for id in subjids:
#import tfrecord and apply encoder

#output of encoder is 4096x1

#import behavioral data for specific sample from csv, append output of encoder

#update svm




def varbatch_svm(dataset: str, batch_size=32, test_size=110; buffer_size=8, lr=1e-3, eps=5e-1, partial=False):
    """ varbatch_svm creates an svm regression model with variable batch size. It takes as inputs the dataset, the 
    batch size, the learning rate, and the threshold for acceptable error
    """
    
    #make sure that you have all necessary variables in necessary forms
    assert isinstance(dataset, str) and len(dataset)
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(test_size, int) and test_size > 0
    assert isinstance(buffer_size, int) and batch_size > 0
    assert isinstance(lr, float) and lr > 0
    assert isinstance(eps, float) and eps > 0
    assert isinstance(partial, bool)
    
    # get dataset path
    dataset_path = _util.get_rel_datasets_path(dataset)
    _util.ensure_dir(dataset_path)
    shape = load_shape(dataset_path) 
    
    #subset data into train and test sets
    train_set = _get_dataset(dataset, batch_size, buffer_size, partial)
    eval_set = _get_dataset(dataset, test_size, buffer_size, partial)

    #define svm inputs vars
    feat= tf.placeholder(dtype=tf.float32,shape=[None, 1])
    label= tf.placeholder(dtype=tf.float32, shape=[None, 1])
    w = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    #define output
    svm_out = tf.add(tf.matmul(feat, w), b)
    #define constants 
    epsilon = tf.constant([eps])
    
    #define loss function: max(0,|w*x+b|-y-eps)
    loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_out, label)), epsilon)))
    
    #define optimizer: gradient descent 
    my_opt = tf.train.GradientDescentOptimizer(learningrate=LR)
    train_step = my_opt.minimize(loss)
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess
        sess.run(init)
        
        train_loss = []
        test_loss = []
        
        #run the function
        for i in range(200):
            rand_index = np.random.choice(len(x_vals_train), size=batch_size)
            X = np.transpose([x_vals_train[rand_index]])
            Y = np.transpose([y_vals_train[rand_index]])
            sess.run(train_step, feed_dict={feat: X, label: Y})
            temp_train_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
            train_loss.append(temp_train_loss)
            temp_test_loss = sess.run(loss, feed_dict={x_data:np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
            test_loss.append(temp_test_loss)
        
        
def apply_encoder(subjid, tfrecordloc, encoder):
    """
    applies autoencoder to subject scans and assignes feature vectors to test and train sets
    """
    featvect=encoder('tfrecordloc\subjid') # how to apply encoder?
    return featvect




# Start main section of code

#TODO figure out which of these lines we actaully need if using Ryan's dataset functions? 
    # Read TFRecord file
    reader = tf.TFRecordReader()
    record_s=glob.glob('*.tfrecords')
    filename_queue = tf.train.string_input_producer(record_S)
    
    _, serialized_example = reader.read(filename_queue)
    # Define features
    read_features = {
            'scan': tf.FixedLenFeature([], dtype=tf.float32), #TODO : is this the right type?
            'behavioral': tf.FizedLenFeature(dtype=tf.float),
            'label': tf.FixedLenFeature(dtype=tf.float32)
            }
    
    # Extract features from serialized data
    read_data = tf.parse_single_example(serialized=serialized_example,features=read_features)
    label=features['label']
    scan=features['scan']
    behav=features['behavioral']
    # Many tf.train functions use tf.train.QueueRunner,
    # so we need to start it before we read
        tf.train.start_queue_runners(sess)
        
    # apply encoder to 'scan', append results to 'behavioral', remove NIH flanker, card sort, picseq, list sort, 
        # pattern from behavioral data if there
    for name, tensor in read_data.items()
        #TODO open most recent saved autoencoder
        
        #TODO remove the decoder portion
        
        #TODO apply to each record
        
        
        #TODO concatenate resulting 4096 feature vector with behavioral features
            



