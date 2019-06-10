"""
Defines a 5D version of Total Variation loss. Inspired by TF's total_variation but handles 3D images.
"""
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def total_variation_5d(images, name=None):
    with ops.name_scope(name, 'total_variation'):
        ndims = images.get_shape().ndims

        if ndims == 5:
            # The input is a batch of images with shape:
            # [batch, height, width, depth, channels].

            # Calculate the difference of neighboring pixel-values.
            # The images are shifted one pixel along the height and width by slicing.
            pixel_dif1 = images[:, 1:, :, :, :] - images[:, :-1, :, :, :]
            pixel_dif2 = images[:, :, 1:, :, :] - images[:, :, :-1, :, :]
            pixel_dif3 = images[:, :, :, 1:, :] - images[:, :, :, :-1, :]

            # Only sum for the last 4 axis.
            # This results in a 1-D tensor with the total variation for each image.
            sum_axis = [1, 2, 3, 4]
        else:
            return tf.image.total_variation(images, name=name)

        # Calculate the total variation by taking the absolute value of the
        # pixel-differences and summing over the appropriate axis.
        tot_var = (
                math_ops.reduce_sum(math_ops.abs(pixel_dif1), axis=sum_axis) +
                math_ops.reduce_sum(math_ops.abs(pixel_dif2), axis=sum_axis) +
                math_ops.reduce_sum(math_ops.abs(pixel_dif3), axis=sum_axis))

    return tot_var
