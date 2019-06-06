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

from data.dataset import get_records, load_shape, _decode
from model.train import _get_dataset, _only_cropped_scan

#TODOS
# Kernel Functions if wanted


def varbatch_svm(dataset: str, batch_size=32, test_size=110; buffer_size=8, LR=1e-3, eps=5e-1, thresh=.005, partial=False):
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
    
    #subset data into train and test sets, randomly shuffle?
        # for now, hardcoade number of files for train/test split:
    testset=np.random.choice(1096,110) #this is a 10% validation split
    test_set = get_dataset_regress(dataset, test_size, buffer_size, partial, testset)
    trainset=np.random.permute(np.delete(np.arange(1:1096), test_set))
    train_set = get_dataset_regress(dataset, batch_size, buffer_size, partial, trainset)

    #define svm inputs vars
    feat= tf.placeholder(dtype=tf.float32,shape=[None, 1])
    label= tf.placeholder(dtype=tf.float32, shape=[None, 1])
    w = tf.Variable(tf.random_normal(shape=[1,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))
    #define output
    svm_out = tf.add(tf.matmul(feat, w), b)
    #define constants 
    epsilon = tf.constant([eps])
    
    #define loss function: |w|**2/2 + C* max(0,|w*x+b|-y-eps); here, C*max computes the mean 
    slackterm= tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_out, label)), epsilon)))
    loss = tf.add(tf.divide(tf.square(w),2),slackterm)
    
    #define optimizer: gradient descent 
    my_opt = tf.train.GradientDescentOptimizer(learningrate=LR)
    train_step = my_opt.minimize(loss)
    init = tf.initialize_all_variables()
    
    #reopen saver
    saver = tf.train.Saver() 
    
    with tf.Session() as sess
        sess.run(init)
                
        #restore best version of model
        bestmodel=get_model() #TODO write function to get best model file path
        saver.restore(sess, bestmodel)
        
        # TODO apply model to each scan, extract 4096x1 feature vector
        featurespart1=apply_encoder(train_set)
        
        # TODO concatenate model feature vector and behavioral feature vector, remove features[6:19] (redundant to fluid intelligence)
        
        train_loss = []
        test_loss = []
        losscrit=thresh+1
        
        #run the function
        while losscrit> thresh:
            rand_index = np.random.choice(len(x_vals_train), size=batch_size)
            X = np.transpose([x_vals_train[rand_index]])
            Y = np.transpose([y_vals_train[rand_index]])
            sess.run(train_step, feed_dict={feat: X, label: Y})
            temp_train_loss = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
            train_loss.append(temp_train_loss)
            temp_test_loss = sess.run(loss, feed_dict={x_data:np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
            test_loss.append(temp_test_loss)
            losscrit=(test_loss[-1]-test_loss[-2])/test_loss[-2]
        
        
def apply_encoder(dataset, encoder): #TODO write function to apply encoder 
    """
    applies autoencoder to subject scans and assignes feature vectors to test and train sets
    """
    featvect=encoder('tfrecordloc\subjid') # how to apply encoder?
    return featvect

def get_model(): 
    #TODO  find most recent model file; should be best model file
    return modelfilepath

def get_dataset_regress(dataset_path: str, batch_size: int, buffer_size: int, shuffle=False, partial=False, indices):
    assert isinstance(batch_size, int) and batch_size > 0
    assert isinstance(buffer_size, int) and buffer_size > 0
    assert isinstance(indices, bool) and indices>0

    records = np.array(get_records(dataset_path, partial))
    records=list(records[indices])
    shape = load_shape(dataset_path)

    return (tf.data.TFRecordDataset(records)
            .map(functools.partial(_decode, shape))
            .batch(batch_size)
            .prefetch(buffer_size))


def main():
    args = _cmd.parse_args_for_callable(varbatch_svm)
    varsArgs = vars(args)
    verbosity = varsArgs.pop('verbosity', _util.DEFAULT_VERBOSITY)

    varbatch_svm(**varsArgs)            



