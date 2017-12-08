from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import sys
import numpy as np
from matplotlib import pyplot as plt

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()


def test1():
    x = tf.matmul([[1, 2], [3, 4]], [[4, 5], [6, 7]])
    y = tf.add(x, 1)
    # z = tf.random_uniform([5, 3])
    z = tf.random_uniform([5, 3])
    print(x)
    print(y)
    print(z)
    n_ele = 1000
    z = tf.random_normal([n_ele])
    print(z)
    plt.hist(z.numpy(), bins=100)
    # plt.plot(z.numpy())
    plt.show()


def f1(x):
    # return x * x
    return tf.multiply(x, x)


def f2(x, y):
    return f1(x) + f1(y)


def test_gradient():
    print(f1(5.0))
    # assert 25 == f(5.0).numpy()
    df1 = tfe.gradients_function(f1)
    print(df1(3.0))
    assert df1(5.0)[0].numpy() == 10

    assert f2(1.0, 2.0).numpy() == 5.0

    df2 = tfe.gradients_function(f2)
    print(df2(1.0, 2.0))


def test_linear_regression():
    """
    Build linear regression with tf.eager
    """

    def pred(input, w, b):
        return input * w + b

    num_examples = 1000
    train_inputs = tf.random_normal([num_examples])
    noise = tf.random_normal([num_examples])
    train_outputs = train_inputs * 3 + 2 + noise

    def loss(w, b):
        error = pred(train_inputs, w, b) - train_outputs
        return tf.reduce_mean(tf.square(error))

    dloss = tfe.gradients_function(loss)

    W = 5.0
    B = 10.0
    lr = 0.01

    print("Initial Loss: {}".format(loss(W, B).numpy()))
    for i in range(200):
        (dW, dB) = dloss(W, B)
        W -= dW * lr
        B -= dB * lr
        if i % 20 == 0:
            print("Loss at step {}: {} with W={}, B={}".format(
                i,
                loss(W, B).numpy(), W.numpy(), B.numpy()))
    print("Final Loss: {}".format(loss(W, B).numpy()))
    print("W: {}, B: {}".format(W.numpy(), B.numpy()))


if __name__ == "__main__":
    test_linear_regression()
