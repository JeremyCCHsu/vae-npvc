from tensorflow.contrib import slim
import numpy as np
import tensorflow as tf

EPSILON = 1e-6

def lrelu(x, leak=0.02, name="lrelu"):
    ''' Leaky ReLU '''
    return tf.maximum(x, leak*x, name=name)


def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        c = np.log(2 * np.pi)
        var = tf.exp(log_var)
        x_mu2 = tf.square(x - mu)   # [Issue] not sure the dim works or not?
        x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, -1)   # keep_dims=True,
        return log_prob


def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    '''
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)
        return tf.reduce_sum(dimwise_kld, -1)

def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))
