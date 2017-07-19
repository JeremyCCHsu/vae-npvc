import tensorflow as tf
from tensorflow.contrib import slim
from util.image import nchw_to_nhwc
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu)
from model.wgan import GradientPenaltyWGAN

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
        for net in ['encoder', 'generator', 'discriminator']:
            assert len(self.arch[net]['output']) > 2
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel'])
            assert len(self.arch[net]['output']) == len(self.arch[net]['stride'])


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
                    tf.zeros_like(slim.flatten(xh))),
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


class VAWGAN(GradientPenaltyWGAN, ConvVAE):
    ''' Conditional on `y` '''
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        with tf.name_scope('SpeakerRepr'):        
            self.y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self._generate = tf.make_template('Generator', self._generator)
        self._encode = tf.make_template('Encoder', self._encoder)
        self._discriminate = tf.make_template('Discriminator', self._discriminator)

        self.generate = self.decode


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
        return x
    

    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        with tf.name_scope('loss'):        

            z = self._generate_noise_with_shape(x)
            z_mu, z_lv = self._encode(x, is_training=self.is_training)
            z_enc = GaussianSampleLayer(z_mu, z_lv)

            z_all = tf.concat([z_enc, z], 0)
            y_all = tf.concat([y, y], 0) if y is not None else None       
            x_ = self._generate(z_all, y_all, is_training=self.is_training)

            xh, x_fake = tf.split(x_, 2)

            x_real = x

            x_all = tf.concat([x_real, x_fake], 0)

            e = tf.random_uniform([tf.shape(x)[0], 1, 1, 1], 0., 1., name='epsilon')
            x_intp = x_real + e * (x_fake - x_real)   # HINT: (1 - a)*A + a*B = A + a(B - A)

            x_all = tf.concat([x_all, x_intp], axis=0)
            y_all = tf.concat([y_all, y], axis=0)

            c_ = self._discriminate(x=x_all, y=y_all)
            c_real, c_fake, c_intp = tf.split(c_, 3)

            with tf.name_scope('loss'):
                gp = self._compute_gradient_penalty(c_intp, x_intp)
                
                # VAE loss
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
                        tf.zeros_like(slim.flatten(xh))
                    )
                )


                loss = dict()
                loss['E_real'] = tf.reduce_mean(c_real)
                loss['E_fake'] = tf.reduce_mean(c_fake)
                loss['W_dist'] = loss['E_real'] - loss['E_fake']
                a = self.arch['training']['alpha']
                loss['l_G'] = - a * loss['E_fake'] + (- logPx + D_KL)
                loss['D_KL'] = D_KL
                loss['logP'] = logPx
                loss['gp'] = gp

                lam = self.arch['training']['lambda']                
                loss['l_D'] = - loss['W_dist'] + lam * gp

                tf.summary.scalar('W_dist', loss['W_dist'])
                tf.summary.scalar('gp', gp)
                tf.summary.scalar('l_G', loss['l_G'] )
                tf.summary.scalar('l_D', loss['l_D'])                
                
                tf.summary.scalar('KL-div', D_KL)
                tf.summary.scalar('logPx', logPx)

                tf.summary.histogram('xh', xh)
                tf.summary.histogram('x', x)

            return loss

