# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 22:57:22 2019

@author: Lynda



"""

import numpy as np
import tensorflow as tf



#TODOS
# Kernel Functions if wanted, 
# formatting & structure for compatibility with network output
# main and/or supporting functions?


# from hcp_config, get subjids and behavioral data attributes

# split subjids into train and tes

# for id in subjids:
#import tfrecord and apply encoder

#output of encoder is 4096x1

#import behavioral data for specific sample from csv, append output of encoder

#update svm




def apply_encoder(subjid, tfrecordloc, encoder):
    """
    applies autoencoder to subject scans and assignes feature vectors to test and train sets
    """
    featvect=encoder('tfrecordloc\subjid') # how to apply encoder?
    return featvect

def varbatch_svm(x_vals_train, x_vals_test, y_vals_train, y_vals_test, batch_size, LR, eps):
    """ varbatch_svm creates an svm regression model with variable batch size. It takes as inputs the features 
    for training data, the features for testing, the labels for training, and the labels for testing, the 
    batch size, the learning rate, and the threshold for acceptable error
    """
    #define placeholder vars
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
    my_opt = tf.train.GradientDescentOptimizer(LR)
    train_step = my_opt.minimize(loss)
    init = tf.initialize_all_variables()
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




# Start main section of code
        
sess = tf.Session()
# get subject ID's and behavioral features from hcp_config
subjid=hcp_config.py
testid=np.random.choice(len(subjID),round(len(subjID)/15)) # choose subjects for 15% test set
batch_size=20; #2% batch size for svm mbgd

for id in subjID:
    # Read TFRecord file
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([str(id)+'.tfrecord'])
    
    _, serialized_example = reader.read(filename_queue)
    # Define features
    read_features = {
            'scan': tf.FixedLenFeature([], dtype=tf.int64),
            'behavioral': tf.FizedLenFeature(dtype=tf.string),
            'label': tf.FixedLenFeature(dtype=tf.float32)
            }
    
    # Extract features from serialized data
    read_data = tf.parse_single_example(serialized=serialized_example,features=read_features)
    
    # Many tf.train functions use tf.train.QueueRunner,
    # so we need to start it before we read
        tf.train.start_queue_runners(sess)
        
    # Print features
    for name, tensor in read_data.items():
        print('{}: {}'.format(name, tensor.eval()))
            



