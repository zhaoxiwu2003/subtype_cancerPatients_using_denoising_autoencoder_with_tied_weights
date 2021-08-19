import os
from abc import ABC

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, optimizers
from utils_cancer_type import *

# gpus = tf.config.experimental.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(gpus[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def my_loop(name, learning_rate=0.001, probes=0.9):
    epoch_number = 500
    batch_size = 32

    data = preprocess_data(name)
    training_data, test_data = splitting_data(data)
    inputs_dim = tf.shape(training_data)[-1]

    name1 = name.strip(".txt")
    weights_name = "Weights_for_" + name1 + "_at_" + str(learning_rate) + "_and_" + str(probes)
    neck_data_name = "Neck_data_for_" + name1 + "_at_" + str(learning_rate) + "_and_" + str(probes)
    """
    1. Add random noise to the training_data. This is similar to the dropout, so do not add it to test_data.
    2. setup datasets from data_with_noise and training_data; training_data as y, data_with_noise as input x;
    data_with_noise==>encoder==>h_neck==>decoder==>X_hat ------> loss <----------data
    """
    noise_shape = tf.reverse(tf.shape(training_data), axis=[0])
    tf.random.set_seed(seed=1)
    noise_counts = tf.ones(noise_shape[-1])
    noise_probes = [probes]
    data_with_noise = tf.math.multiply(
        tf.transpose(adding_noise(noise_shape, [123, 456], noise_counts, noise_probes)), training_data, )
    data_with_noise = tf.cast(data_with_noise, dtype=tf.float32)
    my_dataset = tf.data.Dataset.from_tensor_slices((data_with_noise, training_data)).batch(batch_size)
    # initialize the model
    my_model = MyAEWithTiedWeightsModel(inputs_dim)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    epoch_count = -1
    epoch_loss = 0
    loss_threshold = [np.inf]
    loss_list = []
    test_loss_list = []
    corr_list = []
    test_corr_list = []

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
                loss_list.append(float(loss))
                corr_list.append(float(correlation_coefficient))
        print('name is ', name, '   learning_rate = ', learning_rate, 'probes = ', probes, '   epoch = ', epoch,
              '   train_loss = ', float(loss),'     train_correlation = ', float(correlation_coefficient))

        # test the model in each epoch
        y_test_data = my_model(test_data)
        test_loss = tf.math.reduce_mean(tf.keras.losses.mse(y_test_data, test_data))
        test_correlation_coefficient = tf.reduce_mean(
            tfp.stats.correlation(y_test_data, test_data, sample_axis=1, event_axis=None))
        print(" ")
        print("=" * 200)
        print('name is ', name, '   learning_rate = ', learning_rate, 'probes = ', probes, '   epoch = ', epoch,
              '   test_loss = ', float(test_loss), '     test_correlation = ', float(test_correlation_coefficient))
        print("=" * 200)
        print(" ")
        test_loss_list.append(float(test_loss))
        test_corr_list.append(float(test_correlation_coefficient))

        # set up the early stopping
        if (loss_threshold[-1] - test_loss) / test_loss > 0.001:
            loss_threshold.append(test_loss)
            epoch_loss = epoch_count
        if epoch_count - epoch_loss > 30:
            my_model.save_weights(weights_name)
            data_neck = my_model.encoder(data)
            data_neck1 = data_neck.numpy()
            np.save(neck_data_name, data_neck1)
            my_model.summary()
            print("=" * 50)
            print("Stopped at epoch", epoch_count)
            print("=" * 50)
            return float(loss), float(test_loss), float(correlation_coefficient),\
                   float(test_correlation_coefficient), loss_list, corr_list, test_loss_list, test_corr_list
    my_model.save_weights(weights_name)
    data_neck = my_model.encoder(data)
    data_neck1 = data_neck.numpy()
    np.save(neck_data_name, data_neck1)
    my_model.summary()
    return float(loss), float(test_loss), float(correlation_coefficient), \
           float(test_correlation_coefficient), loss_list, corr_list, test_loss_list, test_corr_list


def main():
    # mirna(1870,441), protein(181,441), gn(20530,441), methy(23782, 441), cna(9176, 441)
    data_sets_names = ["mirna.expr.2017-04-10.txt", "protein.expr.2017-06-27.txt", "gn.expr.2017-04-10.txt",
                       "methy.2017-06-27.txt", "cna.nocnv.nodup.2017-04-10.txt"]
    learning_rate_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    probs_list = [0.9, 0.7, 0.5, 0.3, 0.1]
    with open("result_for_cancer_subtype.txt", "w") as f:
        f.write("name" + "\t" + "learning_rate" + "\t" + "probes" + "\t" + "train_loss" + "\t" + "test_loss" + "\t"
                + "train_corr" + "\t" + "test_corr" + "\n")
    for name in data_sets_names:
        for learning_rate in learning_rate_list:
            for probes in probs_list:
                result = my_loop(name, learning_rate=learning_rate, probes=probes)
                with open("result_for_cancer_subtype.txt", "a") as f:
                    f.write(name + "\t" + str(learning_rate) + "\t" + str(probes) + "\t" + str(result[0]) + "\t" + str(
                        result[1]) + "\t" + str(result[2]) + "\t" + str(result[3])+"\n")
                with open("loss_and_corr_for_all.txt", "a") as ff:
                    ff.write(name + "\t" + str(learning_rate) + "\t" + str(probes) + "\t" + str(result[4])+"\n" +
                             name + "\t" + str(learning_rate) + "\t" + str(probes) + "\t" + str(result[5])+"\n" +
                             name + "\t" + str(learning_rate) + "\t" + str(probes) + "\t" + str(result[6])+"\n" +
                             name + "\t" + str(learning_rate) + "\t" + str(probes) + "\t" + str(result[7])+"\n")


if __name__ == '__main__':
    main()
