# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 02:09:05 2020

@author: xiwuzhao
"""
from abc import ABC
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, Model, Sequential


def preprocess_data(dir_path_file_name):
    # load the txt data into numpy array
    data = np.loadtxt(dir_path_file_name)

    # convert numpy data to tensorflow tensors
    data = tf.convert_to_tensor(np.transpose(data), dtype=tf.float32)

    '''
    data_raw_min = tf.math.reduce_min(data, axis=1)
    data_raw_min = tf.reshape(data_raw_min,(data.shape[0],1))
    eps = 0.00001*tf.ones((data.shape[0],1),dtype=tf.float32)
    data = data - data_raw_min + eps
    data = tf.math.log(data) 
    '''

    # normalize data to range (0-1)
    data_max = tf.math.reduce_max(data, axis=1)
    data_min = tf.math.reduce_min(data, axis=1)
    data_std = tf.math.reduce_std(data, axis=1)
    data_mean = tf.math.reduce_mean(data, axis=1)

    data_max = tf.reshape(data_max, (tf.shape(data)[0], 1))
    data_min = tf.reshape(data_min, (tf.shape(data)[0], 1))
    data_std = tf.reshape(data_std, (tf.shape(data)[0], 1))
    data_mean = tf.reshape(data_mean, (tf.shape(data)[0], 1))

    data = (data - data_mean) / data_max - data_min
    # data = data*2-1

    return data


def adding_noise(noise_shape, noise_seed, noise_counts, probs):
    # adding binomial noise to the data, similar to dropout
    binomial_array = tf.random.stateless_binomial(shape=noise_shape, seed=noise_seed, counts=noise_counts, probs=probs,
                                                  output_dtype=tf.float32)
    return binomial_array


def splitting_data(data):
    x = tf.cast(tf.shape(data)[0], dtype=tf.float32)
    xx = x * 0.7
    x = tf.cast(x, dtype=tf.int32)
    xx = tf.cast(xx, dtype=tf.int32)
    training_data = data[0:xx, :]
    test_data = data[xx:x, :]
    return training_data, test_data


def combine_data():
    """
    1. import data and convert them to tensor
    2. Concatenates them to (441, 250)
    """
    x1 = tf.convert_to_tensor(np.load("Neck_data_for_gn.expr.2017-04-10_at_0.001_and_0.9.npy"), dtype=tf.float32)
    x11 = x1/tf.reshape(tf.reduce_max(x1, axis=1),shape=(-1,1))
    x2 = tf.convert_to_tensor(np.load("NNeck_data_for_cna.nocnv.nodup.2017-04-10_at_0.0001_and_0.9.npy"), dtype=tf.float32)
    x22 = x2/tf.reshape(tf.reduce_max(x2, axis=1),shape=(-1,1))
    x3 = tf.convert_to_tensor(np.load("NNeck_data_for_methy.2017-06-27_at_0.0001_and_0.5.npy"), dtype=tf.float32)
    x33 = x3/tf.reshape(tf.reduce_max(x3, axis=1),shape=(-1,1))
    x4 = tf.convert_to_tensor(np.load("Neck_data_for_mirna.expr.2017-04-10_at_0.0005_and_0.9.npy"), dtype=tf.float32)
    x44 = x4/tf.reshape(tf.reduce_max(x4, axis=1),shape=(-1,1))
    x5 = tf.convert_to_tensor(np.load("Neck_data_for_protein.expr.2017-06-27_at_0.001_and_0.9.npy"), dtype=tf.float32)
    x55 = x5/tf.reshape(tf.reduce_max(x5, axis=1),shape=(-1,1))
    combined_data = tf.concat([x11, x22, x33, x44, x55], 1)
    return combined_data


class TransposeDenseLayer(layers.Layer):
    def __init__(self, units):
        super(TransposeDenseLayer, self).__init__()
        self.b_weights = self.add_weight(name='bias1',
                                         shape=(units,),
                                         initializer='zeros',
                                         trainable=True)

    def call(self, inputs, w_weights):
        z = tf.matmul(inputs, w_weights, transpose_b=True) + self.b_weights
        return z


class MyAEWithTiedWeightsModel(keras.Model, ABC):

    def __init__(self, inputs_dim):
        super(MyAEWithTiedWeightsModel, self).__init__()
        if inputs_dim > 1000:
            self.dense1 = layers.Dense(500, activation="relu")
            self.dense2 = layers.Dense(50, activation="relu")
            self.dense3 = TransposeDenseLayer(500)
            self.dense4 = TransposeDenseLayer(inputs_dim)
            self.inputs_dim = inputs_dim
        else:
            self.dense2 = layers.Dense(50, activation="relu")
            self.dense4 = TransposeDenseLayer(inputs_dim)
            self.inputs_dim = inputs_dim

    def encoder(self, x):
        if self.inputs_dim > 1000:
            # h = tf.nn.relu(self.dense1(x))
            # h_neck = tf.nn.sigmoid(self.dense2(h))
            # h_neck = tf.nn.relu(self.dense2(h))
            h = self.dense1(x)
            h_neck = self.dense2(h)
            return h_neck
        else:
            h_neck = self.dense2(x)
            return h_neck

    def decoder(self, h_neck):
        if self.inputs_dim > 1000:
            out = tf.nn.relu(self.dense3(h_neck, self.dense2.weights[0]))
            out = self.dense4(out, self.dense1.weights[0])
            return out
        else:
            out = self.dense4(h_neck, self.dense2.weights[0])
            return out

    def call(self, inputs, training=None):
        h_neck = self.encoder(inputs)
        x_hat = self.decoder(h_neck)
        return x_hat


class MyAEWithTiedWeightsModel2(keras.Model, ABC):

    def __init__(self):
        super(MyAEWithTiedWeightsModel2, self).__init__()
        self.dense1 = layers.Dense(100, activation="relu")
        self.dense2 = layers.Dense(50, activation="relu")
        self.dense3 = layers.Dense(1, activation="relu")
        self.dense4 = TransposeDenseLayer(50)
        self.dense5 = TransposeDenseLayer(100)
        self.dense6 = TransposeDenseLayer(250)

    def encoder(self, x):
        h = self.dense1(x)
        h = self.dense2(h)
        h_neck = self.dense3(h)

        return h_neck

    def decoder(self, h_neck):
        out = tf.nn.relu(self.dense4(h_neck, self.dense3.weights[0]))
        out = tf.nn.relu(self.dense5(out, self.dense2.weights[0]))
        out = self.dense6(out, self.dense1.weights[0])
        return out

    def call(self, inputs, training=None):
        h_neck = self.encoder(inputs)
        x_hat = self.decoder(h_neck)
        return x_hat
