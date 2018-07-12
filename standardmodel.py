"""Builds multi CNN network only for forward computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf


def _variable_on_cpu(name, shape, para):
    """Helper to create a Variable stored on CPU memory.
    
    Args:
        name: name of the variable
        shape: list of ints
        para: parameter for initializer
    
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        if name == 'weights':
            # initializer = tf.truncated_normal_initializer(stddev=para, dtype=dtype)
            initializer = tf.contrib.layers.xavier_initializer(seed=1)
        else:
            initializer = tf.constant_initializer(para)
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


class Alexnet(object):
    # Build the AlexNet model
    IMAGE_SIZE = 227  # input images size

    def __init__(self, model_flags):

        # Parse input arguments into class variables
        self.NUM_CLASSES = model_flags.num_classes
        self.WEIGHTS_PATH = model_flags.weights_path

    def inference(self, X, reuse=False):

        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID',
                     name='conv1', reuse=reuse)
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2',
                     reuse=reuse)
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')
               
        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3',
                     reuse=reuse)

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4',
                     reuse=reuse)

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5',
                     reuse=reuse)
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        
        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(flattened, 6*6*256, 4096, name='fc6', reuse=reuse)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7', reuse=reuse)

        # 8th Layer: FC and return unscaled activations
        # for tf.nn.softmax_cross_entropy_with_logits
        fc8 = fc(fc7, 4096, self.NUM_CLASSES, relu=False, name='fc8', reuse=reuse)

        return fc7
    
    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of 
        dicts.
        """
        not_load_layers = []
        if self.WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in not_load_layers:
                with tf.variable_scope(op_name, reuse=True):
                    data = weights_dict[op_name]
                    # Biases
                    var = tf.get_variable('biases')
                    session.run(var.assign(data['biases']))
                    # Weights
                    var = tf.get_variable('weights')
                    session.run(var.assign(data['weights']))

        print('Loading the weights is Done.')


class VGG16(object):
    # Build the AlexNet model
    IMAGE_SIZE = 224  # input images size

    def __init__(self, num_classes):

        # Parse input arguments into class variables
        self.NUM_CLASSES = num_classes

    def inference(self, X, reuse=False):
        output = {}
        # 1st Layer: Conv_1-2 (w ReLu) -> Pool
        conv1_1 = conv(X, 3, 3, 64, 1, 1, name='conv1_1', reuse=reuse)
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name='conv1_2', reuse=reuse)
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, name='pool1')
        output['pool1'] = pool1

        # 2nd Layer: Conv_1-2 (w ReLu) -> Pool
        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', reuse=reuse)
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', reuse=reuse)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')
        output['pool2'] = pool2

        # 3rd Layer: Conv_1-3 (w ReLu) -> Pool
        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', reuse=reuse)
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', reuse=reuse)
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', reuse=reuse)
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')
        output['pool3'] = pool3

        # 4th Layer: Conv_1-3 (w ReLu) -> Pool
        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', reuse=reuse)
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', reuse=reuse)
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', reuse=reuse)
        output['conv4_3'] = conv4_3
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')
        output['pool4'] = pool4

        # 5th Layer: Conv_1-3 (w ReLu) -> Pool
        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', reuse=reuse)
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', reuse=reuse)
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', reuse=reuse)
        output['conv5_3'] = conv5_3
        pool5 = max_pool(conv5_3, 2, 2, 2, 2, name='pool5')
        output['pool5'] = pool5

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flat_shape = int(np.prod(pool5.get_shape()[1:]))
        flattened = tf.reshape(pool5, [-1, flat_shape])
        fc6 = fc(flattened, flat_shape, 4096, name='fc6', reuse=reuse)
        output['fc6'] = fc6

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(fc6, 4096, 4096, name='fc7', reuse=reuse)
        output['fc7'] = fc7

        # 8th Layer: FC and return unscaled activations
        # for tf.nn.softmax_cross_entropy_with_logits
        fc8 = fc(fc7, 4096, self.NUM_CLASSES, relu=False, name='fc8', reuse=reuse)
        output['logits'] = fc8

        return output

    def load_initial_weights(self, session, WEIGHTS_PATH):

        not_load_layers = []
        if WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in not_load_layers:
                with tf.variable_scope(op_name, reuse=True):
                    data = weights_dict[op_name]
                    # Biases
                    var = tf.get_variable('biases')
                    session.run(var.assign(data['biases']))
                    # Weights
                    var = tf.get_variable('weights')
                    session.run(var.assign(data['weights']))
                    print('Loaded layer: {}'.format(op_name))

        print('Loading the weights is Done.')

    def init_from_ckpt(self, weight_path=None, preload_layers=[]):
        # This function is called before tf.global_variables_initializer()

        if not weight_path:
            raise ValueError('Please supply the path to a checkpoint of model')
        else:
            wpath = weight_path
            print('Loading the weights of {}'.format(wpath))

        cp_vars = tf.train.list_variables(wpath)
        load_layers = {}
        for var_name, _ in cp_vars:
            tmp_layer = var_name.split('/')[0]
            if tmp_layer not in preload_layers:
                try:
                    tf.get_variable_scope().reuse_variables()
                    load_layers[var_name] = tf.get_variable(var_name)
                except:
                    continue

        print('----------Alreadly loaded variables--------')
        for k in load_layers:
            print(k)

        tf.train.init_from_checkpoint(wpath, load_layers)
        print('Loading the weights is Done.')


"""
Predefine all necessary layer for the CNN 
"""


def conv(x, kernel_height, kernel_width, num_kernels, stride_y, stride_x, name,
         reuse=False, padding='SAME', groups=1):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    print(x.get_shape())

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_height, kernel_width,
                                               input_channels/groups, num_kernels], 1e-1)
        biases = _variable_on_cpu('biases', [num_kernels], 0.0)

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights 
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(value=x, num_or_size_splits=groups, axis=3)
            weight_groups = tf.split(value=weights, num_or_size_splits=groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        bias = tf.nn.bias_add(conv, biases)

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, reuse=False, relu=True):
    with tf.variable_scope(name, reuse=reuse) as scope:

        # Create tf variable for the weights and biases
        # weights = _variable_with_weight_decay('weights', [num_in, num_out], 5e-3, wd)
        weights = _variable_on_cpu('weights', [num_in, num_out], 5e-3)
        biases = _variable_on_cpu('biases', [num_out], 0.1)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, kernel_height, kernel_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kernel_height, kernel_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


