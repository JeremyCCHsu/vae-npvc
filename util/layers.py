# import numpy as np
from math import pi
import tensorflow as tf
# from tensorflow.contrib import slim
from tensorflow.python.ops.init_ops import VarianceScaling

EPSILON = tf.constant(1e-6, dtype=tf.float32)
PI = tf.constant(pi, dtype=tf.float32)

def Layernorm(x, axis, name):
    '''
    Layer normalization (Ba, 2016)
    J: Z-normalization using all nodes of the layer on a per-sample basis.

    Input:
        `x`: channel_first/NCHW format! (or fully-connected)
        `axis`: list
        `name`: must be assigned
    
    Example:
        ```python
        axis = [1, 2, 3]
        x = tf.random_normal([64, 3, 10, 10])
        name = 'D_layernorm'
        ```
    Return:
        (x - u)/s * scale + offset

    Source: 
        https://github.com/igul222/improved_wgan_training/blob/master/tflib/ops/layernorm.py
    '''
    mean, var = tf.nn.moments(x, axis, keep_dims=True)
    n_neurons = x.get_shape().as_list()[axis[0]]
    offset = tf.get_variable(
        name+'.offset',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.zeros_initializer
    )
    scale = tf.get_variable(
        name+'.scale',
        shape=[n_neurons] + [1 for _ in range(len(axis) -1)],
        initializer=tf.ones_initializer
    )
    return tf.nn.batch_normalization(x, mean, var, offset, scale, 1e-5)


def conv2d_nchw_layernorm(x, o, k, s, activation, name):
    '''
    Input:
        `x`: input in NCHW format
        `o`: num of output nodes
        `k`: kernel size
        `s`: stride
    '''
    with tf.variable_scope(name):
        x = tf.layers.conv2d(
            inputs=x,
            filters=o,
            kernel_size=k,
            strides=s,
            padding='same',
            data_format='channels_first',
            name=name,
        )
        x = Layernorm(x, [1, 2, 3], 'layernorm')
        return activation(x)

              
def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
    
def selu_normal(seed=None):
    return VarianceScaling(
        scale=1., mode='fan_in', distribution='normal', seed=seed)

def mu_law_encode_nonlinear(audio, quantization_channels=256):
    '''
    Compress the waveform amplitudes using mu-law non-linearity. 
    NOTE: This mu-law functions as a non-linear function as opposed to 
          quantization.
    '''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.multiply(tf.sign(audio), magnitude, name='mulaw')
        # Quantize signal to the specified number of levels.
        # return tf.to_int32((signal + 1) / 2 * mu + 0.5)
        return signal


def mu_law_decode_nonlinear(output, quantization_channels=256):
    '''
    Uncompress the waveform amplitudes using mu-law non-linearity. 
    NOTE: This mu-law functions as a non-linear function.
    '''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        # signal = 2 * (tf.to_float(output) / mu) - 1
        signal = output
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude


def GumbelSampleLayer(y_mu):
    ''' Create Gumbel(0, 1) variable from Uniform[0, 1] '''
    u = tf.random_uniform(
        minval=0.0,
        maxval=1.0,
        shape=tf.shape(y_mu))
    g = - tf.log(- tf.log(u))
    return y_mu + g


def GumbelSoftmaxLogDensity(y, p, tau):
    # EPS = tf.constant(1e-10)
    k = tf.shape(y)[-1]
    k = tf.cast(k, tf.float32)
    # y = y + EPS
    # y = tf.divide(y, tf.reduce_sum(y, -1, keep_dims=True))
    y = normalize_to_unit_sum(y)
    sum_p_over_y = tf.reduce_sum(tf.divide(p, tf.pow(y, tau)), -1)
    logp = tf.lgamma(k)
    logp = logp + (k - 1) * tf.log(tau)
    logp = logp - k * tf.log(sum_p_over_y)
    logp = logp + sum_p_over_y
    return logp


def normalize_to_unit_sum(x, EPS=1e-10):
    ''' Along the last dim '''
    EPS = tf.constant(EPS, dtype=tf.float32)
    x = x + EPS
    x_sum = tf.reduce_sum(x, -1, keep_dims=True)
    x = tf.divide(x, x_sum)
    return x


def lrelu(x, leak=0.02, name="lrelu"):
    ''' Leaky ReLU '''
    return tf.maximum(x, leak*x, name=name)


def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))


def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        c = tf.log(2. * PI)
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

# Verification by CMU's implementation
# http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
# def gau_kl(pm, pv, qm, qv):
#     """
#     Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
#     Also computes KL divergence from a single Gaussian pm,pv to a set
#     of Gaussians qm,qv.
#     Diagonal covariances are assumed.  Divergence is expressed in nats.
#     """
#     if (len(qm.shape) == 2):
#         axis = 1
#     else:
#         axis = 0
#     # Determinants of diagonal covariances pv, qv
#     dpv = pv.prod()
#     dqv = qv.prod(axis)
#     # Inverse of diagonal covariance qv
#     iqv = 1./qv
#     # Difference between means pm, qm
#     diff = qm - pm
#     return (0.5 *
#             (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
#              + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
#              + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
#              - len(pm)))                     # - N
