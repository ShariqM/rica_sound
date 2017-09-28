import tensorflow as tf
import numpy as np
import pdb
from optparse import OptionParser

import rica_train
import hyperparameters
import utilities
import plotter

parser = OptionParser()
parser.add_option("-l", type="int", dest="load_model_num", default=0,
                  help="Load weights from Model, don't load on 0")
(FLAGS, args) = parser.parse_args()

def initialize():
  hparams = hyperparameters.HyperParameters()
  hparams.load_model_num = FLAGS.load_model_num

  model_num = utilities.get_model_num()
  hparams.logdir  = "logs/%d" % (model_num)
  utilities.setup_tensorboard(hparams.logdir)

  hparams.loaddir = "logs/%d/" % (hparams.load_model_num)
  hparams.savedir = hparams.logdir + "/model.ckpt"

  return hparams

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def compute_snr(x_target, x_hat):
  """Compute the Signal to Noise Ratio (SNR) for the reconstruction."""
  # XXX Assuming data and error are 0-mean
  mean, var = tf.nn.moments(x_hat, axes=[1])
  power_signal = tf.reduce_mean(var)  # Average over batches

  error = x_target - x_hat
  mean, var = tf.nn.moments(error, axes=[1])
  power_noise = tf.reduce_mean(var)  # Average over batches

  return 10 * log10(power_signal / power_noise)

def build_graph(hparams):
  x_shape = [hparams.batch_size, hparams.input_size]
  x = tf.placeholder(tf.float32, shape=x_shape)
  x_target = tf.placeholder(tf.float32, shape=x_shape)

  # AutoEncoder (use tf.contrib.layers.linear)
  r = tf.contrib.layers.linear(x, hparams.num_filters,
                               biases_initializer=None, scope="Analysis")
  x_hat = tf.contrib.layers.linear(r, hparams.num_filters,
                                   biases_initializer=None, scope="Synthesis")

  # loss (use tf.reduce_sum, tf.reduce_mean, tf.square, etc.)
  mse_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x_target - x_hat), axis=1))
  sparsity_loss = hparams.Lambda * tf.reduce_mean(tf.reduce_sum(tf.abs(r), axis=1))
  loss = mse_loss + sparsity_loss
  snr = compute_snr(x_target, x_hat)

  with tf.name_scope("Metrics"):
    mse_loss_summary = tf.summary.scalar("MSE Loss", mse_loss)
    sparse_loss_summary = tf.summary.scalar("Sparsity Loss", sparsity_loss)
    loss_summary = tf.summary.scalar("Loss", loss)
    snr_summary  = tf.summary.scalar("SNR.dB", snr)

  basis_summaries = []
  #for scope in ("Analysis", "Synthesis"):
  #for scope in ("Analysis",):
  for scope in ("Synthesis",):
    filters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)[0]
    basis_buffer = tf.py_func(plotter.get_plot_buf, [filters], tf.string)

    basis_image = tf.image.decode_png(basis_buffer, channels=4)
    basis_image = tf.expand_dims(basis_image, 0)  # make it batched
    basis_summaries.append(
        tf.summary.image('%s Filters' % scope, basis_image, max_outputs=1))
    break

  return (x, x_target, basis_buffer, loss, basis_summaries,
          [mse_loss_summary, sparse_loss_summary, loss_summary, snr_summary])

def main():
  hparams = initialize()
  x, x_target, basis_buffer, loss, basis_summaries, summaries = build_graph(hparams)
  rica_train.run_train(hparams, x, x_target, basis_buffer, loss, basis_summaries, summaries)

if __name__ == '__main__':
  main()
