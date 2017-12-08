from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
"""
Implement simple GAN network using tf graph execution
"""
latent_dim = 100
n_classes = 10
n_features = 784
n_rows = 4
n_cols = 4


def discriminator(x, is_conditional=False, y=None):
    """
    Input:
        - X: real/fake images from real distribution / from generator - shape = [-1, 784]
        - y: condinal label
    Return:
        - probability of X is real/fake (1.0: real, 0.0: fake) - shape = [-1, 1]
    """
    # 1st hidden layer
    if is_conditional and y is not None:
        # y should be one-hot vector of n_classes
        tf.assert_equal(y.shape[1], n_classes)
        x = tf.concat(values=[x, y], axis=1)
        input_dim = n_features + n_classes
    else:
        input_dim = n_features
    d_w1 = tf.get_variable(
        "d_w1",
        shape=[input_dim, 128],
        initializer=tf.random_normal_initializer)
    d_b1 = tf.get_variable(
        "d_b1", shape=[128], initializer=tf.zeros_initializer)
    # print (d_w1, x, d_b1)
    d_h1 = tf.nn.tanh(tf.matmul(x, d_w1) + d_b1)

    # output layer => shape = [1]
    d_w2 = tf.get_variable(
        "d_w2", shape=[128, 1], initializer=tf.random_normal_initializer)
    d_b2 = tf.get_variable("d_b2", shape=[1], initializer=tf.zeros_initializer)
    d_logits = tf.matmul(d_h1, d_w2) + d_b2
    d_probs = tf.nn.sigmoid(d_logits)

    return d_logits, d_probs


def generator(z, is_conditional=False, y=None):
    """
    Input:
        - z: latent variable - size is the same with output
            - shape = [28, 28]
            - other people use shape = [100,]
        - y: condinal label
    Return:
        Generated images: size = [784] (28x28)
    """
    if is_conditional and y is not None:
        # y should be one-hot vector of n_classes
        tf.assert_equal(y.shape[1], n_classes)
        z = tf.concat(values=[z, y], axis=1)  # 1st axis is batch dim
        input_dim = latent_dim + n_classes
    else:
        input_dim = latent_dim
    g_w1 = tf.get_variable(
        "g_w1",
        shape=[input_dim, 128],
        initializer=tf.random_normal_initializer)
    g_b1 = tf.get_variable(
        "g_b1", shape=[128], initializer=tf.zeros_initializer)

    g_w2 = tf.get_variable(
        "g_w2",
        shape=[128, n_features],
        initializer=tf.random_normal_initializer)
    g_b2 = tf.get_variable(
        "g_b2", shape=[n_features], initializer=tf.zeros_initializer)

    g_h1 = tf.nn.tanh(tf.matmul(z, g_w1) + g_b1)
    g_logits = tf.matmul(g_h1, g_w2) + g_b2
    g_probs = tf.sigmoid(g_logits)
    return g_logits, g_probs


def encoder(x, is_conditional=False, y=None):
    """
    Input:
        x: real images
    Return:
        z: values in latent space
    """
    if is_conditional and y is not None:
        # y should be one-hot vector of n_classes
        tf.assert_equal(y.shape[1], n_classes)
        x = tf.concat(values=[x, y], axis=1)  # 1st axis is batch dim
        input_dim = n_features + n_classes
    else:
        input_dim = n_features
    g_w1 = tf.get_variable(
        "e_w1",
        shape=[input_dim, 128],
        initializer=tf.random_normal_initializer)
    g_b1 = tf.get_variable(
        "e_b1", shape=[128], initializer=tf.zeros_initializer)

    g_w2 = tf.get_variable(
        "e_w2",
        shape=[128, latent_dim],
        initializer=tf.random_normal_initializer)
    g_b2 = tf.get_variable(
        "e_b2", shape=[latent_dim], initializer=tf.zeros_initializer)

    h1 = tf.nn.tanh(tf.matmul(x, g_w1) + g_b1)
    e_logits = tf.matmul(h1, g_w2) + g_b2
    e_probs = tf.sigmoid(e_logits)
    return e_probs


def encoder_loss(g_e_x, x, scope="l2_loss"):
    """
    L-2 loss
    """
    with tf.name_scope(scope):
        return tf.norm(g_e_x - x)


def generator_loss(D_G_z, scope="generator_loss"):
    """
    Input:
        - D_G_z: output of D for generated images with latent variable z
        - y: condinal label
    Return:
    """
    with tf.name_scope(scope):
        return tf.reduce_mean(tf.log(1 - D_G_z))
        # return -tf.reduce_mean(tf.log(D_G_z))


def generator_loss_with_logits(D_G_logits, scope="generator_loss_with_logits"):
    """
    Input:
        - y: condinal label
    """
    with tf.name_scope(scope):
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_G_logits, labels=tf.ones_like(D_G_logits)))


def discriminator_loss(D_G_z, D_x, scope="discriminator"):
    """
    Input:
        - D_G_z: output of D for generated images with latent variable z
        - D_x: output of D for real images
        - y: condinal label
    Return:
        - discrimitor loss
    """
    with tf.name_scope(scope):
        return tf.reduce_mean(tf.log(D_x) + tf.log(1. - D_G_z))
        # return tf.reduce_mean(-tf.log(D_x) - tf.log(1. - D_G_z))


def discriminator_loss_with_logits(D_G_logits,
                                   D_x_logits,
                                   scope="discriminator_with_logits",
                                   one_side_smooth_ratio=1.0):
    """
    Input:
        - y: condinal label
    """
    with tf.name_scope(scope):
        # fake losss
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_G_logits, labels=tf.zeros_like(D_G_logits)))
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_x_logits,
                labels=tf.ones_like(D_x_logits) * one_side_smooth_ratio))
        return fake_loss + real_loss


def sample(batch_size, latent_dim):
    return np.random.uniform(-1.0, 1.0, size=[batch_size, latent_dim])
    # return np.random.uniform(0., 1.0, size=[batch_size, latent_dim])


def sample_one_hot_labels(batch_size, n_classes, random=True):
    global n_rows
    n_rows = int(batch_size / n_cols)
    assert batch_size == n_cols * n_rows
    if random:
        labels = np.tile(
            np.random.randint(0, n_classes, size=[n_rows]),
            (n_cols, 1)).T.flatten()
    else:
        labels = np.tile(np.arange(n_rows), (n_cols, 1)).T.flatten()
    temp = np.zeros((batch_size, n_classes))
    temp[np.arange(batch_size), labels] = 1.0
    # print (temp)
    return temp


def plot(samples):
    fig = plt.figure(figsize=(n_cols, n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def train_encoder():
    pass


def train():
    """
    G, D, X, Z
    One step do:
        - Sample m noise sample (Z={z1, .., zm})
        - Sample m real sample X = {x1, .., xm}
        - Generate m fake samples G_z
        - Input generated fakes samples (G_z) to Discriminator 
        => get probability of real/fake D_G_z
        - Input real samples (X) to Discriminator
        => get probability of real/fake D_X (expect to all 1)
    Inputs:
        G: Generator model
        D: Discriminator model
        X: batch of real samples
        Z: batch of generated fake sample
    Returns:
    """
    X = tf.placeholder(
        tf.float32, shape=[None, n_features], name="real_images")
    Z = tf.placeholder(
        tf.float32, shape=[None, latent_dim], name="laten_variables")
    Y = tf.placeholder(tf.float32, shape=[None, n_classes], name="real_labels")
    is_conditional = True

    # train encoder
    en_scope = "encoder"
    with tf.variable_scope(en_scope):
        e_x = encoder(X)

    gen_scope = "generator"
    with tf.variable_scope(gen_scope) as scope:
        g_z_logits, g_z_probs = generator(Z, is_conditional, Y)
        scope.reuse_variables()
        g_e_x_logits, g_e_x_probs = generator(e_x, is_conditional, Y)

    disc_scope = "discriminator"
    with tf.variable_scope(disc_scope) as scope:
        d_x_logits, d_x_probs = discriminator(X, is_conditional,
                                              Y)  # for real images
        scope.reuse_variables()
        d_g_z_logits, d_g_z_probs = discriminator(
            g_z_probs, is_conditional,
            Y)  # input fake images into discriminator
        # d_g_e_x_logits, d_g_e_x_probs = discriminator(
        #     g_e_x_probs, is_conditional, Y)

    # g_loss = generator_loss(d_g_z_probs)
    g_loss = generator_loss_with_logits(d_g_z_logits)

    # d_loss = discriminator_loss(d_g_z_probs, d_x_probs)
    d_loss = discriminator_loss_with_logits(d_g_z_logits, d_x_logits)
    e_loss = encoder_loss(g_e_x_probs, X)

    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope)
    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scope)
    e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=en_scope)

    with tf.name_scope("Generator_Optimizer"):
        g_solver = tf.train.AdamOptimizer().minimize(
            g_loss,
            var_list=g_vars)
    with tf.name_scope("Discriminator_Optimizer"):
        d_solver = tf.train.AdamOptimizer().minimize(
            d_loss,
            var_list=d_vars)

    with tf.name_scope("Encoder_Optimizer"):
        e_solver = tf.train.AdamOptimizer().minimize(
            e_loss,
            var_list=e_vars)

    init = tf.global_variables_initializer()
    log_dir = "logs/train/"
    model_dir = "models/"
    out_dir = "out/tf_gan_graph"

    batch_size = 128
    mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
    summary_ops = tf.summary.merge_all()

    max_iteration = 100000
    log_interval = 1000
    save_interval = 10000
    k = 2
    j = 1
    losses = {}
    d_saver = tf.train.Saver(var_list=d_vars)
    g_saver = tf.train.Saver(var_list=g_vars)
    with tf.Session() as sess:
        sess.run(init)
        tf.train.write_graph(sess.graph.as_graph_def(), log_dir,
                             "train_graph.pbtxt")
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        for i in range(max_iteration):
            real_images, real_labels = mnist.train.next_batch(batch_size)
            # print (real_images.shape)
            for _ in range(k):
                dd_loss, _ = sess.run(
                    [d_loss, d_solver],
                    feed_dict={
                        X: real_images,
                        Z: sample(batch_size, latent_dim),
                        Y: real_labels
                    })
            for _ in range(j):
                gg_loss, _ = sess.run(
                    [g_loss, g_solver],
                    feed_dict={
                        Z: sample(batch_size, latent_dim),
                        # Y: sample_one_hot_labels(batch_size, n_classes)
                        Y: real_labels
                    })
            if i and i % save_interval == 0:
                # save trained variables to disk
                check_point_prefix = os.path.join(
                    model_dir, "generator_discriminator.ckpt")
                saver.save(sess, model_dir, global_step=i)
                losses[i] = {"gen_loss": gg_loss, "disc_loss": dd_loss}

            # if i and i % log_interval == 0:
            if i % log_interval == 0:
                # summary = sess.run(summary_ops)
                # writer.add_summary(summary, global_step=i)
                print(
                    "Iteraion {} th, D loss: {:.4f} -- G loss: {:.4f}".format(
                        i, dd_loss, gg_loss))
                if is_conditional:
                    out_image = 40
                    random = False
                else:
                    out_image = 16
                    random = True
                g_samples = sess.run(
                    g_z_probs,
                    feed_dict={
                        Z: sample(out_image, latent_dim),
                        Y: sample_one_hot_labels(out_image, n_classes, random)
                    })
                # fig = plot(real_images[:16])
                fig = plot(g_samples)
                plt.savefig(
                    "{}/{}.png".format(log_dir, str(i)), bbox_inches='tight')
                plt.close(fig)

    # to train encoder
    # looking for iteration which have best discrimination loss
    best_it = 0
    best_margin = 1
    for i, loss in losses.iteritems():
        loss_margin = abs(loss['disc_loss'] - 0.5)
        if loss_margin < best_margin:
            best_margin = loss_margin
            best_it = i
    print("Iteration {} has the best margin at {} (disc loss: {})".format(
        best_it, best_margin, losses[best_it]['disc_loss']))
    restorer = tf.train.Saver(
        var_list= g_vars + d_vars
    )
    with tf.Session() as sess:
        sess.run(init)
        restorer.restore(sess, )


def gan_model_fn():
    """
    Input:
    Return:
        EstimatorSpec object which include:
            - mode - train mode or evaluation mode
            - prediction op
            - loss op
            - optimizer op for training
            - evaluation op
    """
    pass


def train_input_fn():
    pass


def train_estimator():
    pass


def main():
    train()


if __name__ == "__main__":
    main()