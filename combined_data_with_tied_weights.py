# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 02:41:23 2020

@author: Xiwu Zhao
"""
import os
from abc import ABC
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, optimizers
from utils_cancer_type import *

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def my_loop(learning_rate = 0.001, probes = 0.9):
    epoch_number = 500
    batch_size = 32
    weights_name = "Weights_for_combined_data" + "_at_" + str(learning_rate) + "_and_" + str(probes)
    neck_data_name = "Neck_data_for_combined_data" + "_at_" + str(learning_rate) + "_and_" + str(probes)

    my_model = MyAEWithTiedWeightsModel2()
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    data = combine_data()
    training_data, test_data = splitting_data(data)
    """
    1. Add random noise to the training_data. This is similar to the dropout, so do not add it to test_data.
    2. setup datasets from data_with_noise and training_data; training_data as y, data_with_noise as input x;
    data_with_noise==>encoder==>h_neck==>decoder==>X_hat ------> loss <----------data
    """
    noise_shape = tf.reverse(tf.shape(training_data), axis=[0])
    tf.random.set_seed(seed=1)
    noise_counts = tf.ones(noise_shape[-1])
    noise_probes = probes
    data_with_noise = tf.math.multiply(
        tf.transpose(adding_noise(noise_shape, [123, 456], noise_counts, noise_probes)), training_data, )
    data_with_noise = tf.cast(data_with_noise, dtype=tf.float32)
    my_dataset = tf.data.Dataset.from_tensor_slices((data_with_noise, training_data)).batch(batch_size)

    loss_threshold = [np.inf]
    loss_list1 = []
    test_loss1 = []
    corr_list = []
    test_corr_list = []
    epoch_count = -1
    epoch_loss = 0

    for epoch in range(epoch_number):
        epoch_count += 1
        for step, (x, y) in enumerate(my_dataset):
            with tf.GradientTape() as tape:
                x_hat = my_model(x)
                loss = tf.keras.losses.mse(y, x_hat)
                loss = tf.math.reduce_mean(loss)
                corr = tfp.stats.correlation(x_hat, y, sample_axis=1, event_axis=None)
                correlation_coefficient = tf.math.reduce_mean(corr)
            grads = tape.gradient(loss, my_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
            if step == 0:
                loss_list1.append(float(loss))
                corr_list.append(float(correlation_coefficient))

        print('epoch = ', epoch, '  train_loss = ', float(loss), '  train_correlation = ', float(correlation_coefficient))
        # st the model
        y_test_data = my_model(test_data)
        test_loss = tf.math.reduce_mean(tf.keras.losses.mse(y_test_data, test_data))
        test_correlation_coefficient = tf.reduce_mean(
            tfp.stats.correlation(y_test_data, test_data, sample_axis=1, event_axis=None))
        print("=" * 100)
        print('epoch = ', epoch, '  test_loss = ', float(test_loss), '  test_correlation = ', float(test_correlation_coefficient))
        print("=" * 100)
        test_loss1.append(float(test_loss))
        test_corr_list.append(float(test_correlation_coefficient))

        # set up the early stopping
        if (loss_threshold[-1] - test_loss) / test_loss > 0.001:
            loss_threshold.append(test_loss)
            epoch_loss = epoch_count
        if epoch_count - epoch_loss > 30:
            my_model.save_weights(weights_name)
            data_neck = my_model.encoder(data)
            data_neck1 = data_neck.numpy()
            np.savetxt(neck_data_name, data_neck1)
            my_model.summary()
            print("=" * 50)
            print("Stopped at epoch", epoch_count)
            print("=" * 50)
            return float(loss), float(test_loss), float(correlation_coefficient), float(test_correlation_coefficient)
    my_model.save_weights(weights_name)
    data_neck = my_model.encoder(data)
    data_neck1 = data_neck.numpy()
    np.savetxt(neck_data_name, data_neck1)
    my_model.summary()
    return float(loss), float(test_loss), float(correlation_coefficient), float(test_correlation_coefficient)


def main():
    learning_rate_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    probs_list = [0.9, 0.7, 0.5, 0.3, 0.1]
    with open("final_result_for_cancer_subtype.txt", "w") as f:
        f.write("learning_rate" + "\t" + "probes" + "\t" + "train_loss" + "\t" + "test_loss" + "\t"
                + "train_corr" + "\t" + "test_corr" + "\n")

    for learning_rate in learning_rate_list:
        for probes in probs_list:
            result = my_loop(learning_rate=learning_rate, probes=probes)
            with open("final_result_for_cancer_subtype.txt", "a") as f:
                f.write(str(learning_rate) + "\t" + str(probes) + "\t" + str(result[0]) + "\t" + str(
                    result[1]) + "\t" + str(result[2]) + "\t" + str(result[3])+"\n")


if __name__ == '__main__':
    main()
