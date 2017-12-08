from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import functools
import tensorflow as tf
import os
from abc import abstractmethod
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow.contrib.eager as tfe


"""
Build Gan model using tf eager execution
"""


class Lenet5BaseNet(tfe.Network):

    def __init__(self):
        super(Lenet5BaseNet, self).__init__()
        self._input_shape = self._get_input_shape()
        self.conv1 = self.track_layer(
            tf.layers.Conv2D(32, 5, activation=tf.nn.relu)
        )
        self.conv2 = self.track_layer(
            tf.layers.Conv2D(64, 5, activation=tf.nn.relu)
        )
        self.fc1 = self.track_layer(
            tf.layers.Dense(1024, activation=tf.nn.relu)
        )
        self.fc2 = self.track_layer(
            tf.layers.Dense(self._get_output_shape(), use_bias=True)
        )
        self.dropout = self.track_layer(tf.layers.Dropout(0.5))
        self.max_pool2d = self.track_layer(
            tf.layers.MaxPooling2D(2, 2, padding='SAME')
        )

    # @abstractmethod
    def _get_input_shape(self):
        return [-1, 28, 28, 1]

    # @abstractmethod
    def _get_output_shape(self):
        return [-1, 10]

    def call(self, inputs, training):
        x = tf.reshape(inputs, self._input_shape)
        x = self.conv1(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.max_pool2d(x)
        x = tf.layers.flatten(x)
        x = self.fc1(x)
        if training:
            x = self.dropout(x)
        return x


class GeneratorNet(tfe.Network):

    def __init__(self):
        super(GeneratorNet, self).__init__()
        self

    def _get_input_shape(self):
        return [-1, 28, 28, 1]

    def _get_output_shape(self):
        return [-1, 28, 28, 1]

    def call(self, inputs, training):
        pass


class DiscriminatorNet(tfe.Network):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.fc1 = self.track_layer(
            Lenet5BaseNet()
        )
        self.fc2 = self.track_layer(
            tf.layers.Dense(self._get_output_shape())
        )

    def _get_input_shape(self):
        return [-1, 28, 28, 1]

    def _get_output_shape(self):
        # one class - biniary classification - real (1.0) or fake(0.0)
        return [-1, 1]

    def call(self, inputs, training):
        pass


class GANNet(tfe.Network):

    def __init__(self):
        super(GANNet, self).__init__()


def loss(predictions, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=predictions,
        labels=labels
    ))


def compute_accuracy(predictions, labels):
    return tf.reduce_sum(
        tf.cast(
            tf.equal(
                tf.argmax(predictions, axis=1,
                          output_type=tf.int64),
                tf.argmax(labels, axis=1,
                          output_type=tf.int64)),
            dtype=tf.float32)) / float(predictions.shape[0].value)


def train_one_epoch(model, optimizer, dataset, name, log_interval=None):
    tf.train.get_or_create_global_step()

    def model_loss(labels, inputs):
        predictions = model(inputs, training=True)
        loss_value = loss(predictions, labels)
        tf.contrib.summary.scalar('{}/loss'.format(name), loss_value)
        tf.contrib.summary.scalar("{}/accuracy", compute_accuracy(predictions, labels))
        return loss_value

    for (batch, (inputs, labels)) in enumerate(tfe.Iterator(dataset)):
        with tf.contrib.summary.record_summaries_every_n_global_steps(100):
            batch_model_loss = functools.partial(model_loss, labels, inputs)
            optimizer.minimize(batch_model_loss, global_step=tf.train.get_global_step())
            if log_interval and batch % log_interval == 0:
                print ("Batch #{}\tLoss: {}".format(batch, batch_model_loss()))


def load_data(data_dir):
    data = input_data.read_data_sets(data_dir, one_hot=True)
    train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels))
    test_ds = tf.data.Dataset.from_tensors((data.test.images, data.test.labels))
    return (train_ds, test_ds)


def main_eager():

    batch_size = 10
    log_dir = "./logs/train"

    # tfe.enable_eager_execution()
    # Load the datasets
    (train_ds, test_ds) = load_data(data_dir='../../MNIST_data')
    train_ds = train_ds.shuffle(60000).batch(batch_size)

    dis_net = DiscriminatorNet()
    optimizer = tf.train.AdamOptimizer()

    summary_writer = tf.contrib.summary.create_summary_file_writer(log_dir, flush_millis=10000)

    if tf.contrib.eager.in_graph_mode():
        pass
        # tf.train.write_graph(dis_net.graph, log_dir, "discriminator.pbtxt")


if __name__ == "__main__":
    main_eager()
