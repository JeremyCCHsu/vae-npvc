import tensorflow as tf
from tensorflow.contrib import slim
from util.image import nchw_to_nhwc
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu)
# from model.wgan import GradientPenaltyWGAN

class ConvVAE(object):
    def __init__(self, arch, is_training=False):
        '''
        Variational auto-encoder implemented in 2D convolutional neural nets
        Input:
            `arch`: network architecture (`dict`)
            `is_training`: (unused now) it was kept for historical reasons (for `BatchNorm`)
        '''
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        with tf.name_scope('SpeakerRepr'):
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self._generate = tf.make_template(
            'Generator',
            self._generator)

        self._encode = tf.make_template(
            'Encoder',
            self._encoder)

        self.generate = self.decode  # for VAE-GAN extension


    def _sanity_check(self):
        for net in ['encoder', 'generator']:
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel']) == len(self.arch[net]['stride'])


    def _unit_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim])
            embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
        return embeddings


    def _merge(self, var_list, fan_out, l2_reg=1e-6):
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=None,
            activation_fn=None):
            for var in var_list:
                x = x + slim.fully_connected(var)
        return slim.bias_add(x)


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        return embeddings

    def _encoder(self, x, is_training=None):
        net = self.arch['encoder']
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu,
                name='Conv2d-{}'.format(i)
            )
        x = slim.flatten(x)
        z_mu = tf.layers.dense(x, self.arch['z_dim'])
        z_lv = tf.layers.dense(x, self.arch['z_dim'])
        return z_mu, z_lv

    def _generator(self, z, y, is_training=None):
        net = self.arch['generator']
        h, w, c = net['hwc']

        if y is not None:
            y = tf.nn.embedding_lookup(self.y_emb, y)
            x = self._merge([z, y], h * w * c)
        else:
            x = z

        x = tf.reshape(x, [-1, c, h, w])  # channel first
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = tf.layers.conv2d_transpose(x, o, k, s,
                padding='same',
                data_format='channels_first',
            )
            if i < len(net['output']) -1:
                x = Layernorm(x, [1, 2, 3], 'ConvT-LN{}'.format(i))
                x = lrelu(x)
            else:
                x = tf.nn.tanh(x) # not 100% necessary
        return x


    def loss(self, x, y):
        with tf.name_scope('loss'):
            z_mu, z_lv = self._encode(x)
            z = GaussianSampleLayer(z_mu, z_lv)
            xh = self._generate(z, y)

            D_KL = tf.reduce_mean(
                GaussianKLD(
                    slim.flatten(z_mu),
                    slim.flatten(z_lv),
                    slim.flatten(tf.zeros_like(z_mu)),
                    slim.flatten(tf.zeros_like(z_lv)),
                )
            )
            logPx = tf.reduce_mean(
                GaussianLogDensity(
                    slim.flatten(x),
                    slim.flatten(xh),
                    tf.zeros_like(slim.flatten(xh))),  # TODO exp -tf.log(1/9) + 
            )

        loss = dict()
        loss['G'] = - logPx + D_KL
        loss['D_KL'] = D_KL
        loss['logP'] = logPx

        tf.summary.scalar('KL-div', D_KL)
        tf.summary.scalar('logPx', logPx)

        tf.summary.histogram('xh', xh)
        tf.summary.histogram('x', x)
        return loss

    def encode(self, x):
        z_mu, _ = self._encode(x)
        return z_mu

    def decode(self, z, y):
        xh = self._generate(z, y)
        return nchw_to_nhwc(xh)
