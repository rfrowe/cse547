import os

import numpy as np
import tensorboard_logger as _tboard
import tensorflow as tf
from tensorflow.contrib.factorization import WALSModel

import utils.utility as _util
import train.train_test_utils as _train_utils

_logger = _util.get_logger(__file__)


def reco(sess, inp, code, label, epsilon, train_dataset, dev_dataset, lr, weights_path):

    # Initialize hyperparameters
    # TODO: Proper tuning_threshold strategy, or is there a better stopping condition?
    # TODO: Grid search for reg_l2 tuning?  Currently only tune factor_dim
    factor_dim = 0
    reg_l2 = 0.1
    factor_loss_thresh = 1e-6
    tuning_thresh = 1e-6

    # Ratings matrix dimensions
    n_items = _train_utils.dataset_iter_len(sess, train_dataset.make_one_shot_iterator().get_next())
    n_users_train = 877
    n_users_dev = 110
    n_users_test = 110

    '''Placeholder labels
    label = np.random.randn(n_users_train + n_users_dev + n_users_test, 1)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    '''
    
    label_train = label[1:n_users_train+1, -1]
    label_dev = label[n_users_train+1 : n_users_train+1 + n_users_dev+1, -1]
    label_test = label[n_users_train+1 + n_users_dev+1 : -1, -1]

    # Rating matrix
    # TODO: Random placeholder data for now.  Rating matrix must include all train/dev/test
    #       data.  Each row represents a user, and each column represents a feature. The label
    #       is to be included in the last feature column, with dev/test set labels removed.
    rating_matrix = np.random.randn(n_users_train + n_users_dev + n_users_test, n_items)
    
    input_tensor = tf.convert_to_tensor(rating_matrix, dtype=tf.float32)
    input_tensor = tf.contrib.layers.dense_to_sparse(input_tensor)

    # Tune model using increasing latent factor matrix dimension
    losscrit = np.inf
    while losscrit > tuning_thresh:

        factor_dim += 1
        
        # Weighted alternating least squares model (causes deprecation warning)
        model = WALSModel(n_users_train + n_users_dev + n_users_test, n_items, factor_dim,
                          regularization = reg_l2,
                          row_weights = None,
                          col_weights = None)

        # Retrieve row and column factors
        users_factor = model.row_factors[0]
        items_factor = model.col_factors[0]

        # Initialize training
        row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
        col_update_op = model.update_col_factors(sp_input=input_tensor)[1]
        sess.run(model.initialize_op)
        sess.run(model.worker_init)

        # Update latent factor matrices via Alternating Least Squares until matrix decomposition converges
        u_factor_old = users_factor.eval(session=sess)
        i_factor_old = items_factor.eval(session=sess)
        factor_loss = np.inf
        while factor_loss > factor_loss_thresh:
            sess.run(model.row_update_prep_gramian_op)
            sess.run(model.initialize_row_update_op)
            sess.run(row_update_op)
            sess.run(model.col_update_prep_gramian_op)
            sess.run(model.initialize_col_update_op)
            sess.run(col_update_op)
            
            u_factor_new = users_factor.eval(session=sess)
            i_factor_new = items_factor.eval(session=sess)
            factor_loss = max(np.linalg.norm(u_factor_new - u_factor_old),
                              np.linalg.norm(i_factor_new - i_factor_old))
            
            u_factor_old = u_factor_new
            i_factor_old = i_factor_new

        # Predictions
        pred_fun = tf.matmul(users_factor, items_factor, transpose_b=True)
        pred = sess.run(pred_fun)
        pred_train = pred[1:n_users_train+1, -1]
        pred_dev = pred[n_users_train+1 : n_users_train+1 + n_users_dev+1, -1]
        pred_test = pred[n_users_train+1 + n_users_dev+1 : -1, -1]

        # Performance
        loss_fun = tf.math.reduce_sum(tf.math.square(tf.abs(pred - label))) + tf.nn.l2_loss(users_factor) + tf.nn.l2_loss(items_factor)
        losscrit = sess.run(loss_fun)
        train_loss = sess.run(tf.reduce_mean(tf.abs(pred_train - label_train)))
        dev_loss = sess.run(tf.reduce_mean(tf.abs(pred_dev - label_dev)))
        test_loss = sess.run(tf.reduce_mean(tf.abs(pred_test - label_test)))
