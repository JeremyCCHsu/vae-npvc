import pdb
# from tensorflow.contrib import slim
import tensorflow as tf
from util.layers import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu

# TODO:  
#  1. Multi-stream?
#  2. Separate MLP & CNN?
#    (or we can also just use CNN by viewing MLP input as D-channels)
#  3. Conditional input as an option during __init__?


class MLPcVAE(object):
    '''
    Conditional Variational Auto-encoder implemented in multi-layer perceptron
    APIs:
        z = encode(x)
        xh = decode(z, y)
    Notation:
        shape: 
            `b`: batch_size, 
            `c`: indicates the dimension
    '''

    def __init__(self, arch, is_training=False):
        self.arch = arch
        self.is_training = is_training
        self._decode = tf.make_template('Decoder', self._generator)
        self._encode = tf.make_template('Encoder', self._encoder)

    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=tf.contrib.keras.regularizers.l2(1e-3),
            )
        return embeddings

    def _encoder(self, x, is_training):
        ''' Expects 2D inputs (b, c)
        Input:
            `x`: shape=[b, c]
        Return:
            `z_mu`: mean vector of `z`
            `z_lv`: log var of `z`
        '''
        for o in self.arch['encoder']['output']:
            x = tf.layers.dense(x, units=o, activation=lrelu)
            # x = tf.layers.batch_normalization(x, training=is_training)
        z_mu = tf.layers.dense(x, units=self.arch['z_dim'], name='z_mu')
        z_lv = tf.layers.dense(x, units=self.arch['z_dim'], name='z_lv')
        return z_mu, z_lv

    def _generator(self, z, y, is_training):
        '''
        Input:
            z: shape=[b, c]
            y: speaker label; shape=[b,], dtype=int64
        Return:
            xh: reconstructed version of `x` (the input to the VAE)
        '''
        self.speaker_repr = self._l2_regularized_embedding(
            n_class=self.arch['y_dim'],
            h_dim=self.arch['yemb_dim'],
            scope_name='y_embedding',
            var_name='y_emb'
        )

        c = tf.nn.embedding_lookup(self.speaker_repr, y)
        x = tf.concat([z, c], -1)
        for o in self.arch['decoder']['output']:
            x = tf.layers.dense(x, units=o, activation=lrelu)            
            # x = tf.layers.batch_normalization(x, training=is_training)
        return tf.layers.dense(x, units=self.arch['x_dim'], name='xh')

    def loss(self, x, y):
        '''
        Args:
            x: shape=[s, b, c]
            y: shape=[s, b]
        Returns:
            a `dict` of losses
        '''
        z_mu, z_lv = self._encode(x, is_training=self.is_training)
        z = GaussianSampleLayer(z_mu, z_lv)
        xh = self._decode(z, y, is_training=self.is_training)

        with tf.name_scope('loss'):
            with tf.name_scope('E_log_p_x_zy'):
                L_x = -1.0 * tf.reduce_mean(
                    GaussianLogDensity(x, xh, tf.zeros_like(x)),
                )
            with tf.name_scope('D_KL_z'):
                L_z = tf.reduce_mean(
                    GaussianKLD(
                        z_mu, z_lv,
                        tf.zeros_like(z_mu), tf.zeros_like(z_lv)
                    )
                )
            loss = {
                'L_x': L_x,
                'L_z': L_z,
            }

        tf.summary.scalar('L_x', L_x)
        tf.summary.scalar('L_z', L_z)
        return loss

    def encode(self, x):
        z_mu, z_lv = self._encode(x, is_training=False)
        return z_mu

    def decode(self, z, y, tanh=False):
        xh = self._decode(z, y, is_training=False)
        return xh
