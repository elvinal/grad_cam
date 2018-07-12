"""Builds the DenseNet network."""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

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


class DenseNet161(object):
    # Build the DenseNet model
    IMAGE_SIZE = 224    # input images size

    def __init__(self, num_classes):

        # Parse input arguments into class variables
        self.net_depth = 161
        self.nlayers = [6, 12, 36, 24]
        self.growth_rate = 48
        self.bn_size = 4
        self.compressed = 0.5
        self.num_classes = num_classes

    def inference(self, X, is_training=False, keep_prob=1.0):

        output = {}
        # First Convluation (224x224)
        x = conv(X, 7, 2*self.growth_rate, 2, pad=3, name='conv0')
        x = BatchNorm(x, is_training, name='norm0')
        x = Relu(x)
        print(x.shape)
        x = max_pool(x, 3, 2, pad=1)
        print(x.shape)
        output['pool0'] = x

        # DenseBlock 1 (56x56)
        x = DenseBlock(x, self.nlayers[0], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock1')
        x = Transition(x, self.compressed, is_training, keep_prob, name='transition1')
        print(x.shape)
        output['transition1'] = x

        # DenseBlock 2 (28x28)
        x = DenseBlock(x, self.nlayers[1], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock2')
        x = Transition(x, self.compressed, is_training, keep_prob, name='transition2')
        print(x.shape)
        output['transition2'] = x

        # DenseBlock 3 (14x14)
        x = DenseBlock(x, self.nlayers[2], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock3')
        x = Transition(x, self.compressed, is_training, keep_prob, name='transition3')
        print(x.shape)
        output['transition3'] = x

        # DenseBlock 4 (7x7)
        x = DenseBlock(x, self.nlayers[3], self.bn_size, self.growth_rate, is_training,
                       keep_prob, name='denseblock4')
        x = BatchNorm(x, is_training, name='norm5')
        x = Relu(x)
        output['norm5'] = x
        x = avgpool(x, 7, 1, padding='VALID')
        print(x.shape)
        x = tf.squeeze(x, [1, 2])
        print(x.shape)
        output['gobalpool'] = x

        # classifier
        x = fc(x, self.num_classes, name='fc5')
        output['logits'] = x

        return output
    
    def load_initial_weights(self, session, WEIGHTS_PATH, wBNm=True):

        not_load_layers = ['fc5']
        if wBNm:
            wBNm_key = ' '
        else:
            wBNm_key = 'moving'
        st = time.time()
        if WEIGHTS_PATH == 'None':
            raise ValueError('Please supply the path to a pre-trained model')

        print('Loading the weights of pre-trained model')

        # load the weights into memory
        weights_dict = np.load(WEIGHTS_PATH, encoding='bytes').item()

        out_layers = []
        # merge all assign ops, just run once can obtain low overhead of time.
        assign_ops = []
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            if wBNm_key in op_name:
                continue
            layer = op_name.split('/')[0]
            # Check if the layer is one of the layers that should be reinitialized
            if layer not in not_load_layers:
                data = weights_dict[op_name]
                tf.get_variable_scope().reuse_variables()
                try:
                    # Biases
                    var = tf.get_variable(op_name)
                    assign_ops.append(var.assign(data))
                    # print("Loaded layer: {}".format(op_name))
                except:
                    print('Not Load Layer: {}'.format(op_name))
                    continue
                out_layers.append(layer)
        session.run(assign_ops)
        prt_strs = ["{} : {}".format(k, out_layers.count(k)) for k in sorted(set(out_layers))]
        for pstr in prt_strs:
            print(pstr)

        print('Loading the weights is Done in {:.4f}s.'.format(time.time() - st))


"""
Predefine all necessary layer for CNN
"""


def DenseBlock(x, nlayers, bn_size, growth_rate, is_training, keep_prob, name):

    with tf.variable_scope(name):
        for i in range(nlayers):
            x = Denselayer(x, bn_size, growth_rate, is_training, keep_prob, name='denselayer{}'.format(i+1))

    return x


def Denselayer(x, bn_size, growth_rate, is_training, keep_prob, name):
    with tf.variable_scope(name):
        # bottleneck layer
        nx = BatchNorm(x, is_training, 'norm_1')
        nx = Relu(nx)
        nx = conv(nx, 1, bn_size*growth_rate, 1, 'conv_1')
        # norm layer
        nx = BatchNorm(nx, is_training, 'norm_2')
        nx = Relu(nx)
        nx = conv(nx, 3, growth_rate, 1, 'conv_2', pad=1)
        nx = dropout(nx, keep_prob)
        nx = tf.concat([x, nx], axis=3)

    return nx


def Transition(x, compressed, is_training, keep_prob, name):
    in_kernels = x.shape.as_list()[-1]
    # print(in_kernels*compressed)
    out_kernels = np.floor(in_kernels*compressed).astype(np.int32)
    with tf.variable_scope(name):
        x = BatchNorm(x, is_training, 'norm')
        x = Relu(x)
        x = conv(x, 1, out_kernels, 1, 'conv')
        x = dropout(x, keep_prob)
        x = avgpool(x, 2, 2)

    return x


def conv(x, kernel_size, num_kernels, stride_size, name, pad=0,
         with_bias=False, reuse=False, padding='VALID'):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
    # print(x.get_shape())

    x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_size, stride_size, 1],
                                         padding=padding)

    with tf.variable_scope(name, reuse=reuse):
        # Create tf variables for the weights and biases of the conv layer
        weights = _variable_on_cpu('weights', [kernel_size, kernel_size,
                                               input_channels, num_kernels], 1e-1)

        # Apply convolution function
        conv = convolve(x, weights)

        if with_bias:
            # Add biases
            biases = _variable_on_cpu('biases', [num_kernels], 0.0)
            conv = tf.nn.bias_add(conv, biases)

        return conv


def BatchNorm(x, is_training, name):
    return tf.contrib.layers.batch_norm(x, decay=0.9, epsilon=1e-5, scale=True,
                                        is_training=is_training, fused=True,
                                        zero_debias_moving_mean=False, scope=name)


def Relu(x):
    return tf.nn.relu(x)


def fc(x, num_out, name, reuse=False,
       relu=False, batch_norm=False, is_training=False):
    num_in = x.shape.as_list()[-1]
    with tf.variable_scope(name, reuse=reuse) as scope:

        # Create tf variable for the weights and biases
        weights = _variable_on_cpu('weights', [num_in, num_out], 1e-1)
        biases = _variable_on_cpu('biases', [num_out], 1.0)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if batch_norm:
            # Adds a Batch Normalization layer
            act = tf.contrib.layers.batch_norm(act, center=True, scale=True,
                                               trainable=True, is_training=is_training,
                                               reuse=reuse, scope=scope)

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, kernel_size, stride_size, pad=0, padding='VALID'):
    x = tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding)


def avgpool(x, kernel_size, stride_size, padding='VALID'):
    return tf.nn.avg_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1],
                          padding=padding)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


