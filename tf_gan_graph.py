from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import math
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
"""
Implementation of simple GAN network using tf graph execution
"""
latent_dim = 100
n_classes = 10
n_features = 784
n_rows = 4
n_cols = 4
train_log_dir = "logs/train/"
explorer_log_dir = "logs/explorer/"
model_dir = "models/"
out_dir = "out/tf_gan_graph"

batch_size = 128
max_iteration = 100000
log_interval = 1000
save_interval = 10000
k = 2
j = 1

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)


def discriminator(x, is_conditional=False, y=None):
  """
    Input:
        - X: real/fake images from real distribution / from generator - shape = [-1, 784]
        - y: use label if is_conditional==True
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
      "d_w1", shape=[input_dim, 128], initializer=tf.random_normal_initializer)
  d_b1 = tf.get_variable("d_b1", shape=[128], initializer=tf.zeros_initializer)
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
      "g_w1", shape=[input_dim, 128], initializer=tf.random_normal_initializer)
  g_b1 = tf.get_variable("g_b1", shape=[128], initializer=tf.zeros_initializer)

  g_w2 = tf.get_variable(
      "g_w2", shape=[128, n_features], initializer=tf.random_normal_initializer)
  g_b2 = tf.get_variable(
      "g_b2", shape=[n_features], initializer=tf.zeros_initializer)

  g_h1 = tf.nn.tanh(tf.matmul(z, g_w1) + g_b1)
  g_logits = tf.matmul(g_h1, g_w2) + g_b2
  g_probs = tf.sigmoid(g_logits)
  return g_logits, g_probs


def encoder(x, is_conditional=True, y=None):
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
      "e_w1", shape=[input_dim, 128], initializer=tf.random_normal_initializer)
  g_b1 = tf.get_variable("e_b1", shape=[128], initializer=tf.zeros_initializer)

  g_w2 = tf.get_variable(
      "e_w2", shape=[128, latent_dim], initializer=tf.random_normal_initializer)
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
      - y: label
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


def plot(samples, n_rows, n_cols, niter=None):
  assert len(samples) == n_cols * n_rows
  fig = plt.figure(figsize=(n_cols, n_rows))
  if niter is not None:
    plt.suptitle("Iteration {}".format(niter))
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


def train(is_conditional=False):
  """
    G, D, X, Z
    One step do:
        - Sample m noise sample (Z={z1, .., zm})
        - Sample m real sample X = {x1, .., xm}
        - Generate m fake samples G_z
        - Input generated fakes samples (G_z) to Discriminator 
        => get probability of real/fake D_G_z (expect to all 0.0)
        - Input real samples (X) to Discriminator
        => get probability of real/fake D_X (expect to all 1.0)
    Inputs:
        G: Generator model
        D: Discriminator model
        X: batch of real samples
        Z: batch of generated fake sample
    Returns:
  """
  X = tf.placeholder(tf.float32, shape=[None, n_features], name="real_images")
  Z = tf.placeholder(
      tf.float32, shape=[None, latent_dim], name="laten_variables")
  Y = tf.placeholder(tf.float32, shape=[None, n_classes], name="real_labels")

  # train encoder
  en_scope = "encoder"
  with tf.variable_scope(en_scope):
    e_x = encoder(X, is_conditional, Y)

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
        g_z_probs, is_conditional, Y)  # input fake images into discriminator

  g_loss = generator_loss_with_logits(d_g_z_logits)
  
  d_loss = discriminator_loss_with_logits(d_g_z_logits, d_x_logits)
  e_loss = encoder_loss(g_e_x_probs, X)

  summary_g_loss = tf.summary.scalar("Generator Loss", g_loss)
  summary_d_loss = tf.summary.scalar("Discriminator Loss", d_loss)
  summary_e_loss = tf.summary.scalar("Encoder Loss", e_loss)

  g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope)
  d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=disc_scope)
  e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=en_scope)

  with tf.name_scope("Generator_Optimizer"):
    g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)
  with tf.name_scope("Discriminator_Optimizer"):
    d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
  with tf.name_scope("Encoder_Optimizer"):
    e_solver = tf.train.AdamOptimizer().minimize(e_loss, var_list=e_vars)

  init = tf.global_variables_initializer()

  losses = {}
  d_saver = tf.train.Saver(
      var_list=d_vars, max_to_keep=int(max_iteration / save_interval))
  g_saver = tf.train.Saver(
      var_list=g_vars, max_to_keep=int(max_iteration / save_interval))

  summary_ops = tf.summary.merge([summary_d_loss, summary_g_loss])
  with tf.Session() as sess:
    sess.run(init)
    tf.train.write_graph(sess.graph.as_graph_def(), train_log_dir,
                         "train_graph.pbtxt")
    writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    for i in range(max_iteration):
      real_images, real_labels = mnist.train.next_batch(batch_size)
      # print (real_images.shape)
      for _ in range(k):
        dd_loss, _, summary = sess.run(
            [d_loss, d_solver, summary_ops],
            feed_dict={
                X: real_images,
                Z: sample(batch_size, latent_dim),
                Y: real_labels
            })
      if (i % (log_interval // 2) == 0):
        writer.add_summary(summary, global_step=i)
      for _ in range(j):
        gg_loss, _, summary = sess.run(
            [g_loss, g_solver, summary_ops],
            feed_dict={
                X: real_images,
                Z: sample(batch_size, latent_dim),
                # Y: sample_one_hot_labels(batch_size, n_classes)
                Y: real_labels
            })
      if (i % (log_interval // 2) == 0):
        writer.add_summary(summary, global_step=i)
      if i and i % save_interval == 0:
        # save trained variables to disk
        g_saver.save(
            sess,
            os.path.join(model_dir, "generator", "generator.ckpt"),
            global_step=i)
        d_saver.save(
            sess,
            os.path.join(model_dir, "discriminator", "discriminator.ckpt"),
            global_step=i)
        losses[i] = {"gen_loss": gg_loss, "disc_loss": dd_loss}

      if i % log_interval == 0:
        print("Iteraion {} th, D loss: {:.4f} -- G loss: {:.4f}".format(
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
        fig = plot(g_samples, n_rows, n_cols, niter=i)
        plt.savefig(
            "{}/{}.png".format(train_log_dir, str(i)), bbox_inches='tight')
        plt.close(fig)



# ---------------------- TRAIN ENCODER -----------------
# looking for iteration which have best discrimination loss
  best_gen_it = 0
  best_margin = 1
  for i, loss in losses.items():
    loss_margin = abs(loss['disc_loss'] - 0.5)
    if loss_margin < best_margin:
      best_margin = loss_margin
      best_gen_it = i
  print("Iteration {} has the best margin at {} (disc loss: {})".format(
      best_gen_it, best_margin, losses[best_gen_it]['disc_loss']))

  # best_gen_it = 80000

  g_restorer = tf.train.Saver(var_list=g_vars)
  d_restorer = tf.train.Saver(var_list=d_vars)
  e_saver = tf.train.Saver(
      var_list=e_vars, max_to_keep=int(max_iteration / save_interval))

  summary_ops = tf.summary.merge([summary_e_loss])
  with tf.Session() as sess:
    sess.run(init)
    g_restorer.restore(
        sess,
        os.path.join(model_dir, "generator",
                     "generator.ckpt-{}".format(best_gen_it)))
    d_restorer.restore(
        sess,
        os.path.join(model_dir, "discriminator",
                     "discriminator.ckpt-{}".format(best_gen_it)))

    best_e_loss = np.Inf
    best_en_it = 0
    for i in range(max_iteration):
      real_images, real_labels = mnist.train.next_batch(batch_size)
      # print (real_images.shape)
      ee_loss, _, summary = sess.run(
          [e_loss, e_solver, summary_ops],
          feed_dict={
              X: real_images,
              Y: real_labels
          })
      if (i % (log_interval // 2) == 0):
        writer.add_summary(summary, global_step=i)
      if i and i % save_interval == 0:
        # save trained variables to disk
        e_saver.save(
            sess,
            os.path.join(model_dir, "encoder", "encoder.ckpt"),
            global_step=i)        
        if ee_loss < best_e_loss:
          best_e_loss = ee_loss
          best_en_it = i
      
      if i % log_interval == 0:        
        print("Iteraion {} th, Encoder loss: {:.4f}".format(i, ee_loss))
        # n_out_image = 16
        n_out_image = real_images.shape[0]
        recontructed_samples = sess.run(
            g_e_x_probs,
            feed_dict={
                X: real_images[:out_image],
                Y: real_labels[:out_image]
            })
        # fig = plot(real_images[:16])
        fig = plot(recontructed_samples, n_rows, n_cols, niter=i)
        plt.savefig(
            "{}/{}_recontructed.png".format(train_log_dir, str(i)),
            bbox_inches='tight')
        plt.close(fig)

        fig = plot(real_images[:out_image], n_rows, n_cols, niter=i)
        plt.savefig(
            "{}/{}_original.png".format(train_log_dir, str(i)),
            bbox_inches='tight')
        plt.close(fig)
  print("Iteration {} has the best encoder loss at {}".format(
      best_en_it, best_e_loss))
  return (best_gen_it, best_en_it)

from functools import partial


def linear_interpolation_in_latent_space_batch_mode(z1,
                                                    z2,
                                                    n_steps,
                                                    x_z_1=None,
                                                    x_z_2=None):
  inter_fn = partial(linear_interpolation_in_latent_space, n_steps=n_steps)
  elems = (z1, z2, x_z_1, x_z_2)
  return tf.map_fn(
      fn=lambda elem: inter_fn(z1=elem[0], z2=elem[1], x_z_1=elem[2], x_z_2=elem[3]),
      elems=elems, dtype=(tf.float32)
  )
  # return tf.reshape(x_z_temps, shape=[batch_size, len(x_z_temps), -1], name="intermediate_values")


def linear_interpolation_in_latent_space(z1,
                                         z2,
                                         n_steps,
                                         x_z_1=None,
                                         x_z_2=None):
  # linear interpolation
  x_z_temps = []
  step_vec = tf.div(z2 - z1, n_steps)
  if x_z_1 is not None:
    x_z_temps.append(x_z_1)

  for i in range(n_steps):
    z_temp = z1 + (i) * step_vec
    _, x_z_temp = generator(
        tf.expand_dims(z_temp, axis=0), is_conditional=False)
    x_z_temps.append(tf.squeeze(x_z_temp, axis=0))
  if x_z_2 is not None:
    x_z_temps.append(x_z_2)
  return tf.reshape(x_z_temps, shape=[len(x_z_temps), -1])


def plot_rescontructed_images(reconstructed_images, n_rows):
  return plot(reconstructed_images, n_rows)


def sample_pair_images(n_pairs=5):
  x1 = []
  x2 = []
  y1 = []
  y2 = []
  for _ in range(n_pairs):
    real_images, real_labels = mnist.train.next_batch(batch_size=2)
    x1.append(real_images[0])
    x2.append(real_images[1])

    y1.append(real_labels[0])
    y2.append(real_labels[1])
  return (np.asarray(x1), np.asarray(x2)), (np.asarray(y1), np.asarray(y2))


def explore_latent_space(gen_best_it, en_best_it):
  is_conditional = False

  y1 = tf.placeholder(tf.float32, shape=[None, n_classes], name="y1")
  y2 = tf.placeholder(tf.float32, shape=[None, n_classes], name="y2")
  x1 = tf.placeholder(tf.float32, shape=[None, n_features], name="x1")
  x2 = tf.placeholder(tf.float32, shape=[None, n_features], name="x2")
  
  en_scope = "encoder"
  with tf.variable_scope(en_scope, reuse=tf.AUTO_REUSE) as scope:
    z1 = encoder(x1, is_conditional, y1)
    z2 = encoder(x2, is_conditional, y2)

  explore_scope = "explorer"
  gen_scope = "generator"
  n_pairs = 5
  n_steps = 15
  print(z1.shape, z2.shape)
  with tf.name_scope(explore_scope):
    with tf.variable_scope(gen_scope, reuse=tf.AUTO_REUSE) as scope:
      x_z_intermediates = linear_interpolation_in_latent_space_batch_mode(
          z1,
          z2,
          n_steps=n_steps,
          x_z_1=x1,
          x_z_2=x2,
      )
      print(x_z_intermediates)

  g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=gen_scope)
  e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=en_scope)

  init = tf.global_variables_initializer()
  summary_ops = tf.summary.merge_all()

  g_restorer = tf.train.Saver(var_list=g_vars)
  e_restorer = tf.train.Saver(var_list=e_vars)
  with tf.Session() as sess:
    writer = tf.summary.FileWriter(explorer_log_dir, sess.graph)

    g_restorer.restore(
        sess,
        os.path.join(model_dir, "generator",
                     "generator.ckpt-{}".format(gen_best_it)))
    e_restorer.restore(
        sess,
        os.path.join(model_dir, "encoder",
                     "encoder.ckpt-{}".format(en_best_it)))
    (x1_reals, x2_reals), (y1_reals, y2_reals) = sample_pair_images(n_pairs)
    x_z_intermediates_reals = sess.run(
        x_z_intermediates,
        feed_dict={
            x1: x1_reals,
            x2: x2_reals,
            y1: y1_reals,
            y2: y2_reals
        })
    print("intermediates values: {}".format(x_z_intermediates_reals.shape))
    full_dims = x_z_intermediates_reals.shape
    n_rows = full_dims[0]
    n_cols = full_dims[1]
    flatten_samples = np.reshape(x_z_intermediates_reals, [n_rows * n_cols, -1])
    plot(flatten_samples, n_rows, n_cols)
    plt.savefig(
            "{}/interpolate.png".format(explorer_log_dir),
            bbox_inches='tight')

    plt.show()


def main():
  is_conditional = True
  # is_conditional = False
  best_gen_it, best_en_it = train(is_conditional)  
  # best_gen_it, best_en_it = (50000, 90000)

  best_it_file = "best_it.pkl"
  import pickle
  import os
  with open(best_it_file, "wb+") as f:
    pickle.dump((best_gen_it, best_en_it), f)  
  if os.path.isfile(best_it_file):
      (best_gen_it, best_en_it) = pickle.load(open(best_it_file, "rb"))
  if not is_conditional:
    explore_latent_space(
        gen_best_it=best_gen_it,
        en_best_it=best_en_it,
    )


if __name__ == "__main__":
  main()
