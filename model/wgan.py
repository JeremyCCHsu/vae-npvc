import tensorflow as tf
from tensorflow.contrib import slim
from util.image import nchw_to_nhwc
from util.layers import (GaussianLogDensity, Layernorm, conv2d_nchw_layernorm,
                         lrelu)


class GradientPenaltyWGAN(object):
    '''
    Wasserstein GAN with Gradient Penalty (conditional version)  
    '''

    def __init__(self, arch, is_training=False):
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        self.y_emb = self._speaker_embedding('AttributeEmbedding')

        self._generate = tf.make_template('Generator', self._generator)
        self._discriminate = tf.make_template('Discriminator', self._discriminator)


    def _sanity_check(self):
        for net in ['generator', 'discriminator']:
            assert len(self.arch[net]['output']) > 2
            assert len(self.arch[net]['output']) == len(self.arch[net]['kernel'])
            assert len(self.arch[net]['output']) == len(self.arch[net]['stride'])


    def _speaker_embedding(self, name):
        with tf.name_scope(name):
            y_emb = self._l2_regularized_embedding(
                self.arch['y_dim'] - 1,
                self.arch['y_emb_dim'],
                'y_embedding')
        return tf.concat(
            [tf.zeros([1, self.arch['y_emb_dim']]), y_emb],
            axis=0)


    # def _unit_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
    #     with tf.variable_scope(scope_name):
    #         embeddings = tf.get_variable(
    #             name=var_name,
    #             shape=[n_class, h_dim])
    #         embeddings = tf.nn.l2_normalize(embeddings, dim=-1, name=var_name+'normalized')
    #     return embeddings


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                initializer=tf.random_normal_initializer,
                regularizer=slim.l2_regularizer(1e-3))
        return embeddings


    def make_y_feature_map(self, y, embedding):#, hwc=[96, 96, 1]):
        ''' 
        FIXME: Designed for HW3 
        (but I think this should be outside the model) 
        '''
        y = tf.nn.embedding_lookup(embedding, y)  # [b, a, y_emb]
        # y = tf.reduce_sum(y, axis=1)
        return y


    def _generator(self, z, y, is_training):
        ''' Can be conditioned on `y` or not '''
        subnet = self.arch['generator']
        n_layer = len(subnet['output'])
        h, w, c = subnet['hwc']

        if y is not None:
            y_emb = self.make_y_feature_map(y, self.y_emb)
            x = tf.concat([z, y_emb], 1)
        else:
            x = z

        with slim.arg_scope([slim.batch_norm], scale=True, scope='BN',
            updates_collections=None, decay=0.9, epsilon=1e-5, is_training=is_training):
            x = slim.fully_connected(
                x,
                h * w * c,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu,
            )
            x = tf.reshape(x, [-1, c, h, w])
            for i in range(n_layer):
                # Don't apply BN for the last layer of G
                activation = None if i == (n_layer -1) else lrelu
                normalizer = None if i == (n_layer -1) else slim.batch_norm
                x = slim.conv2d_transpose(
                    x,
                    subnet['output'][i],
                    subnet['kernel'][i],
                    subnet['stride'][i],
                    activation_fn=activation,
                    normalizer_fn=normalizer,
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    data_format='NCHW',
                )
        return tf.nn.tanh(x) #, x    # x, logit


    def _discriminator(self, x, y, is_training=None):
        ''' "No critic batch normalization," pp. 6 '''
        net = self.arch['discriminator']
        h, w, c = self.arch['hwc']

        if y is not None:
            y_emb = self.make_y_feature_map(y, self.y_emb)
            # y_emb = slim.fully_connected(y_emb, h * w * c, activation_fn=None)
            # y_emb = tf.reshape(y_emb, [-1, c, h, w])
            # x = tf.concat([x, y_emb], 1)

        # Radford: [do] not applying batchnorm to the discriminator input layer
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(x, o, k, s, lrelu, name='Conv2d-{}'.format(i))

        # Don't apply BN for the last layer
        x = slim.flatten(x)
        x = x if y is None else tf.concat([x, y_emb], 1)

        x = slim.fully_connected(x, 512, activation_fn=None)
        x = Layernorm(x, [1], 'layernorm-last')
        x = lrelu(x)
        x = slim.fully_connected(x, 1, activation_fn=None)
        return x  # no explicit `sigmoid`


    def _generate_noise_with_shape(self, x):
        ''' iW-GAN used Gaussian noise '''
        with tf.name_scope('Z'):
            z = tf.random_normal([tf.shape(x)[0], self.arch['z_dim']], name='z') 
        return z


    def _compute_gradient_penalty(self, J, x, scope_name='GradientPenalty'):
        ''' Gradient Penalty
        Input:
            `J`: the loss
            `x`: shape = [b, c, h, w]
        '''
        with tf.name_scope(scope_name):
            grad = tf.gradients(J, x)[0]  # as the output is a list, [0] is needed
            grad_square = tf.square(grad)
            grad_squared_norm = tf.reduce_sum(grad_square, axis=[1, 2, 3])
            grad_norm = tf.sqrt(grad_squared_norm)
            # penalty = tf.square(tf.nn.relu(grad_norm - 1.)) # FIXME: experimental
            penalty = tf.square(grad_norm - 1.)
        return tf.reduce_mean(penalty)


    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        z = self._generate_noise_with_shape(x)

        y_all = None if y is None else tf.concat([y, y, y], 0)
        x_real = x
        x_fake = self._generate(z, y, is_training=self.is_training)

        e = tf.random_uniform([tf.shape(x)[0], 1, 1, 1], 0., 1., name='epsilon')
        x_intp = x_real + e * (x_fake - x_real)   # HINT: (1 - a)*A + a*B = A + a(B - A)

        c_ = self._discriminate(
            x=tf.concat([x_real, x_fake, x_intp], 0),
            y=y_all,
            is_training=self.is_training
        )
        c_real, c_fake, c_intp = tf.split(c_, 3)


        with tf.name_scope('loss'):
            gp = self._compute_gradient_penalty(c_intp, x_intp)

            l_mean = -1. * tf.reduce_mean(
                GaussianLogDensity(
                    x=c_real,
                    mu=tf.zeros_like(c_real),
                    log_var=tf.zeros_like(c_real))
            )
            # D: max E[D(x)] - E[D(G(z))]  => L_D: minus
            # G: max E[D(G(z))]            => L_G: minus
            lam = self.arch['training']['lambda']
            loss = dict()
            loss['E_real'] = tf.reduce_mean(c_real)
            loss['E_fake'] = tf.reduce_mean(c_fake)
            loss['W_dist'] = loss['E_real'] - loss['E_fake']
            loss['l_G'] = - loss['E_fake']
            loss['l_D'] = - loss['W_dist'] + lam * gp + l_mean

        # # For summaries
        # with tf.name_scope('Summary'):
            tf.summary.scalar('E_real', loss['E_real'])
            tf.summary.scalar('E_fake', loss['E_fake'])
            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('gp', gp)
            tf.summary.scalar('W_dist', loss['W_dist'])
            tf.summary.histogram('c_real', c_real)
            tf.summary.histogram('c_fake', c_fake)
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake))
            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss

    def discriminate(self, x, y):
        '''
        To estimate the EMD, we need D to assign a score per sample.
        *The batches can be of different size
        '''
        return self._discriminate(x, y, is_training=False)

    def generate(self, z, y):
        ''' `y` can be `None` '''
        # z = self._generate_noise_with_shape(y)
        # return self._generate(z, y, is_training=False)
        if y is None:
            xh = self._generate(z, y=None, is_training=False)
        else:
            xh = self._generate(z, y, is_training=False)
        return nchw_to_nhwc(xh)


# 

# class LeastSquareGAN(GradientPenaltyWGAN):
#     ''' 
#     Same structure, diff loss function
#     NOTE: unsuccessful trial on MLSS-HW3, ending up with mode collapsing. 
#     '''
#     def loss(self, x, y):
#         '''
#         Return:
#             `loss`: a `dict` for the trainer (keys must match)
#         '''
#         z = self._generate_noise_with_shape(x)

#         y_all = None if y is None else tf.concat([y, y], 0)
#         x_real = x
#         x_fake = self._generate(z, y, is_training=self.is_training)

#         c_ = self._discriminate(
#             x=tf.concat([x_real, x_fake], 0),
#             y=y_all,
#             is_training=self.is_training
#         )
#         c_real, c_fake = tf.split(c_, 2)


#         with tf.name_scope('loss'):
#             loss = dict()
#             loss['E_real'] = tf.reduce_mean(tf.square(c_real - 1.)) / 2.
#             loss['E_fake'] = tf.reduce_mean(tf.square(c_fake + 1.)) / 2.
#             loss['l_G'] =    tf.reduce_mean(tf.square(c_fake - 0.)) / 2.
#             loss['l_D'] = loss['E_real'] + loss['E_fake']

#             # tf.summary.scalar('E_real', loss['E_real'])
#             # tf.summary.scalar('E_fake', loss['E_fake'])
#             tf.summary.scalar('l_D', loss['l_D'])
#             tf.summary.scalar('l_G', loss['l_G'])
#             tf.summary.histogram('y', y)
#             tf.summary.histogram('c_real', c_real)
#             tf.summary.histogram('c_fake', c_fake)
#             tf.summary.histogram('x_real', slim.flatten(x_real))
#             tf.summary.histogram('x_fake', slim.flatten(x_fake))
#             tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
#             tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)
#         return loss


class WassersteinGAN_GPC(GradientPenaltyWGAN):  
    def _generator(self, z, y, is_training):
        ''' Can be conditioned on `y` or not '''
        subnet = self.arch['generator']
        n_layer = len(subnet['output'])
        h, w, c = subnet['hwc']

        x = z
        with slim.arg_scope([slim.batch_norm], scale=True, scope='BN',
            updates_collections=None, decay=0.9, epsilon=1e-5, is_training=is_training):
            x = slim.fully_connected(
                x,
                h * w * c,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu,
            )
            x = tf.reshape(x, [-1, c, h, w])
            for i in range(n_layer):
                # Don't apply BN for the last layer of G
                activation = None if i == (n_layer -1) else lrelu
                normalizer = None if i == (n_layer -1) else slim.batch_norm
                x = slim.conv2d_transpose(
                    x,
                    subnet['output'][i],
                    subnet['kernel'][i],
                    subnet['stride'][i],
                    activation_fn=activation,
                    normalizer_fn=normalizer,
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    data_format='NCHW',
                )
        # return tf.nn.tanh(x) #, x    # x, logit
            x = tf.nn.tanh(x)
        
            if y is not None:
                # y_emb = self.make_y_feature_map(y, self.y_emb)
                # x = tf.concat([z, y_emb], 1)
                x = tf.concat([x, y], 1)

            for i in range(len(subnet['output2'])):
                # Don't apply BN for the last layer of G
                activation = None if i == (n_layer -1) else lrelu
                normalizer = None if i == (n_layer -1) else slim.batch_norm
                x = slim.conv2d_transpose(
                    x,
                    subnet['output2'][i],
                    subnet['kernel2'][i],
                    activation_fn=activation,
                    normalizer_fn=normalizer,
                    weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                    data_format='NCHW',
                )
        return tf.nn.tanh(x)

    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        z = self._generate_noise_with_shape(x)

        y_all = None if y is None else tf.concat([y, y, y], 0)
        x_real = x
        x_fake = self._generate(z, y, is_training=self.is_training)

        e = tf.random_uniform([tf.shape(x)[0], 1, 1, 1], 0., 1., name='epsilon')
        x_intp = x_real + e * (x_fake - x_real)   # HINT: (1 - a)*A + a*B = A + a(B - A)

        c_ = self._discriminate(
            x=tf.concat([x_real, x_fake, x_intp], 0),
            y=None,  # J: only this
            is_training=self.is_training
        )
        c_real, c_fake, c_intp = tf.split(c_, 3)


        with tf.name_scope('loss'):
            gp = self._compute_gradient_penalty(c_intp, x_intp)

            l_mean = -1. * tf.reduce_mean(
                GaussianLogDensity(
                    x=c_real,
                    mu=tf.zeros_like(c_real),
                    log_var=tf.zeros_like(c_real))
            )
            # D: max E[D(x)] - E[D(G(z))]  => L_D: minus
            # G: max E[D(G(z))]            => L_G: minus
            lam = self.arch['training']['lambda']
            loss = dict()
            loss['E_real'] = tf.reduce_mean(c_real)
            loss['E_fake'] = tf.reduce_mean(c_fake)
            loss['W_dist'] = loss['E_real'] - loss['E_fake']
            loss['l_G'] = - loss['E_fake']
            loss['l_D'] = - loss['W_dist'] + lam * gp + l_mean

            # For summaries
            tf.summary.scalar('E_real', loss['E_real'])
            tf.summary.scalar('E_fake', loss['E_fake'])
            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('gp', gp)
            tf.summary.scalar('W_dist', loss['W_dist'])
            tf.summary.histogram('c_real', c_real)
            tf.summary.histogram('c_fake', c_fake)
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake))
            tf.summary.audio('x_real', slim.flatten(x_real), 16000, 5)
            tf.summary.audio('x_fake', slim.flatten(x_fake), 16000, 5)

            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss



class CramerGAN(GradientPenaltyWGAN):
    '''
    Cramer GAN and W-GAN are both computationally costly because they train D for k times.
    cf. LSGAN: 4.5 step per sec
         CGAN:  <1 step per sec
    '''

    def _discriminator(self, x, y, is_training):
        ''' "No critic batch normalization," pp. 6 '''
        net = self.arch['discriminator']

        if y is not None:
            y_emb = self.make_y_feature_map(y, self.y_emb)

        # Radford: [do] not applying batchnorm to the discriminator input layer
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(x, o, k, s, lrelu, name='Conv2d-{}'.format(i))

        x = slim.flatten(x)
        x = x if y is None else tf.concat([x, y_emb], 1)

        # Don't apply BN for the last layer
        x = tf.layers.dense(x, net['h_dim'], name='FC0')
        x = Layernorm(x, [1], 'layernorm-last')
        x = lrelu(x)
        x = tf.layers.dense(x, net['h_dim'], name='FC1')
        return x  # no explicit `sigmoid`

    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        with tf.name_scope('Z'):
            # z = tf.random_normal(
            #     [2 * tf.shape(x)[0], self.arch['z_dim']],
            #     name='z'
            # )
            z = tf.concat(
                [self._generate_noise_with_shape(x),
                 self._generate_noise_with_shape(x)],
                axis=0
            )

        x_real = x
        x_fake = self._generate(z, tf.concat([y, y], 0), is_training=self.is_training)
        x_fake1, x_fake2 = tf.split(x_fake, 2)

        with tf.name_scope('Interpolation'):
            e = tf.random_uniform([tf.shape(x)[0], 1, 1, 1], 0., 1., name='epsilon')
            x_intp = x_real + e * (x_fake1 - x_real)   # HINT: (1 - a)*A + a*B = A + a(B - A)

        x_list = [x_real, x_fake1, x_fake2, x_intp]
        y_all = None if y is None else tf.concat([y] * len(x_list), 0)
        
        h_ = self._discriminate(
            x=tf.concat(x_list, 0),
            y=y_all,
            is_training=self.is_training
        )
        h_real, h_fake1, h_fake2, h_intp = tf.split(h_, len(x_list))
        # output from _dis is 256-dim

        def _critic(xi, xg):
            with tf.name_scope('Critic'):
                return tf.norm(xi - xg, axis=1) - tf.norm(xi, axis=1)

        with tf.name_scope('loss'):
            
            Ef_r = tf.reduce_mean(_critic(h_real, h_fake2)) 
            Ef_f = tf.reduce_mean(_critic(h_fake1, h_fake2))
            Ef_i = tf.reduce_mean(_critic(h_intp, h_fake2))

            gp = self._compute_gradient_penalty(Ef_i, x_intp)

            lam = self.arch['training']['lambda']
            loss = dict()

            loss['l_G'] = Ef_r - Ef_f
            loss['l_D'] = - loss['l_G'] + lam * gp
            loss['gp'] = gp

            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('l_G', loss['l_G'])
            tf.summary.scalar('gp', loss['gp'])            
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake1))
            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake1), max_outputs=4)

        return loss

# 


class CramerGANAngularInput(CramerGAN):
    def make_y_feature_map(self, y, embedding):  # , hwc=[96, 96, 1]):
        ''' 
        FIXME: Designed for HW3 
        (but I think this should be outside the model) 
        '''
        y = tf.nn.embedding_lookup(embedding, y)  # [b, a, y_emb]
        y = tf.reduce_sum(y, axis=1)
        return tf.cos(y)

    def _generate_noise_with_shape(self, x):
        ''' iW-GAN used Gaussian noise '''
        with tf.name_scope('Z'):
            z = tf.random_uniform(
                shape=[tf.shape(x)[0], self.arch['z_dim']],
                minval=-3.14159,
                maxval=3.14159,
                name='z')
        return tf.cos(z)

          
class BiWGAN(CramerGANAngularInput):
    ''' Conditional on `y` '''

    def __init__(self, arch, is_training=False):
        super(BiWGAN, self).__init__(arch, is_training)
        self._encode = tf.make_template('Encoder', self._encoder)

    def _encoder(self, x, is_training):
        # Radford: [do] not applying batchnorm to the discriminator input layer
        net = self.arch['discriminator']
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu, name='Conv2d-{}'.format(i))

        x = slim.flatten(x)
        x = tf.layers.dense(x, self.arch['z_dim'])
        return x

    def _discriminator(self, x, y, z, is_training):
        ''' "No critic batch normalization," pp. 6 '''
        net = self.arch['discriminator']

        if y is not None:
            y_emb = self.make_y_feature_map(y, self.y_emb)

        # Radford: [do] not applying batchnorm to the discriminator input layer
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu, name='Conv2d-{}'.format(i))

        x = slim.flatten(x)
        x = tf.concat([x, z], axis=1)
        x = x if y is None else tf.concat([x, y_emb], 1)

        x = tf.layers.dense(x, 512)
        x = Layernorm(x, [1], 'layernorm-last')
        x = lrelu(x)
        x = tf.layers.dense(x, 1)
        return x  # no explicit `sigmoid`

    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        z = self._generate_noise_with_shape(x)
        z_real = self._encode(x, is_training=self.is_training)

        y_all = None if y is None else tf.concat([y, y, y], 0)
        x_real = x
        x_fake = self._generate(z, y, is_training=self.is_training)

        e = tf.random_uniform([tf.shape(x)[0], 1, 1, 1],
                              0., 1., name='epsilon')
        # HINT: (1 - a)*A + a*B = A + a(B - A)
        x_intp = x_real + e * (x_fake - x_real)

        e = tf.reshape(e, [-1, 1])
        z_intp = z + e * (z - z_real)
        z_all = tf.concat([z_real, z, z_intp], axis=0)
        c_ = self._discriminate(
            x=tf.concat([x_real, x_fake, x_intp], 0),
            y=y_all,
            z=z_all,
            is_training=self.is_training
        )
        c_real, c_fake, c_intp = tf.split(c_, 3)

        with tf.name_scope('loss'):
            gp = self._compute_gradient_penalty(c_intp, x_intp)

            l_mean = -1. * tf.reduce_mean(
                GaussianLogDensity(
                    x=c_real,
                    mu=tf.zeros_like(c_real),
                    log_var=tf.zeros_like(c_real))
            )
            # D: max E[D(x)] - E[D(G(z))]  => L_D: minus
            # G: max E[D(G(z))]            => L_G: minus
            lam = self.arch['training']['lambda']
            loss = dict()
            loss['E_real'] = tf.reduce_mean(c_real)
            loss['E_fake'] = tf.reduce_mean(c_fake)
            loss['W_dist'] = loss['E_real'] - loss['E_fake']
            loss['l_G'] = - loss['E_fake']
            loss['l_D'] = - loss['W_dist'] + lam * gp + l_mean
            loss['gp'] = gp

        # # For summaries
        # with tf.name_scope('Summary'):
            tf.summary.scalar('E_real', loss['E_real'])
            tf.summary.scalar('E_fake', loss['E_fake'])
            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('gp', gp)
            tf.summary.scalar('W_dist', loss['W_dist'])
            tf.summary.histogram('c_real', c_real)
            tf.summary.histogram('c_fake', c_fake)
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake))
            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss
