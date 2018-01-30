import pdb
from tensorflow.contrib import slim
import tensorflow as tf
from util.layer import GaussianLogDensity, GaussianKLD, \
    GaussianSampleLayer, lrelu

class VAWGAN(object):
    '''
      VC-GAN
    = CVAE-CGAN
    = Convolutional Variational Auto-encoder
      with Conditional Generative Adversarial Net
    '''
    def __init__(self, arch, is_training=False):
        self.arch = arch
        self._sanity_check()
        self.is_training = is_training

        with tf.name_scope('SpeakerRepr'):        
            self.y_emb = self._unit_embedding(
                self.arch['y_dim'],
                self.arch['z_dim'],
                'y_embedding')

        self._generate = tf.make_template(
            'Generator',
            self._generator)
        self._discriminate = tf.make_template(
            'Discriminator',
            self._discriminator)
        self._encode = tf.make_template(
            'Encoder',
            self._encoder)


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
        ''' 
        Note: Don't apply BN on this because 'y' 
              tends to be the same inside a batch.
        '''
        x = 0.
        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=fan_out,
            weights_regularizer=slim.l2_regularizer(l2_reg),
            normalizer_fn=None,
            activation_fn=None):
            for var in var_list:
                x = x + slim.fully_connected(var)
        x = slim.bias_add(x)
        return x


    def _l2_regularized_embedding(self, n_class, h_dim, scope_name, var_name='y_emb'):
        with tf.variable_scope(scope_name):
            embeddings = tf.get_variable(
                name=var_name,
                shape=[n_class, h_dim],
                regularizer=slim.l2_regularizer(1e-6))
        return embeddings


    def _encoder(self, x, is_training):
        n_layer = len(self.arch['encoder']['output'])
        subnet = self.arch['encoder']

        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            decay=0.9, epsilon=1e-5,  # [TODO] Test these hyper-parameters
            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i in range(n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])
                    tf.summary.image(
                        'down-sample{:d}'.format(i),
                        tf.transpose(x[:, :, :, 0:3], [2, 1, 0, 3]))

        x = slim.flatten(x)

        with slim.arg_scope(
            [slim.fully_connected],
            num_outputs=self.arch['z_dim'],
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            normalizer_fn=None,
            activation_fn=None):
            z_mu = slim.fully_connected(x)
            z_lv = slim.fully_connected(x)
        return z_mu, z_lv


    def _generator(self, z, y, is_training):
        ''' In this version, we only generate the target, so `y` is useless '''
        subnet = self.arch['generator']
        n_layer = len(subnet['output'])
        h, w, c = subnet['hwc']

        y = tf.nn.embedding_lookup(self.y_emb, y)

        x = self._merge([z, y], subnet['merge_dim'])
        x = lrelu(x)
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            decay=0.9, epsilon=1e-5,
            is_training=is_training):

            x = slim.fully_connected(
                x,
                h * w * c,
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu)

            x = tf.reshape(x, [-1, h, w, c])

            with slim.arg_scope(
                [slim.conv2d_transpose],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                for i in range(n_layer -1):
                    x = slim.conv2d_transpose(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i]
                        # normalizer_fn=None
                        )

                # Don't apply BN for the last layer of G
                x = slim.conv2d_transpose(
                    x,
                    subnet['output'][-1],
                    subnet['kernel'][-1],
                    subnet['stride'][-1],
                    normalizer_fn=None,
                    activation_fn=None)

                # pdb.set_trace()
                logit = x
                x = tf.nn.tanh(logit)
        return x, logit


    def _discriminator(self, x, is_training):
    # def _discriminator(self, x, is_training):
        ''' Note: In this version, `y` is useless '''
        subnet = self.arch['discriminator']
        n_layer = len(subnet['output'])

        # y_dim = self.arch['y_dim']
        # h, w, _ = self.arch['hwc']
        # y_emb = self._l2_regularized_embedding(y_dim, h * w, 'y_embedding_disc_in')
        # y_vec = tf.nn.embedding_lookup(y_emb, y)
        # y_vec = tf.reshape(y_vec, [-1, h, w, 1])

        intermediate = list()
        intermediate.append(x)

        # x = tf.concat(3, [x, y_vec])   # inject y into x
        with slim.arg_scope(
            [slim.batch_norm],
            scale=True, scope='BN',
            updates_collections=None,
            decay=0.9, epsilon=1e-5,
            is_training=is_training):
            with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
                normalizer_fn=slim.batch_norm,
                activation_fn=lrelu):

                # Radford: [do] not applying batchnorm to the discriminator input layer
                x = slim.conv2d(
                    x,
                    subnet['output'][0],
                    subnet['kernel'][0],
                    subnet['stride'][0],
                    normalizer_fn=None)
                intermediate.append(x)
                for i in range(1, n_layer):
                    x = slim.conv2d(
                        x,
                        subnet['output'][i],
                        subnet['kernel'][i],
                        subnet['stride'][i])
                    intermediate.append(x)
                    tf.summary.image(
                        'upsampling{:d}'.format(i),
                        tf.transpose(x[:, :, :, 0:3], [2, 1, 0, 3]))

        # Don't apply BN for the last layer
        x = slim.flatten(x)
        h = slim.flatten(intermediate[subnet['feature_layer'] - 1])

        # y_vec = tf.nn.embedding_lookup(self.y_emb, y)

        # x = self._merge([x], subnet['merge_dim'])
        # x = lrelu(x)

        # x = slim.fully_connected(
        #     x,
        #     subnet['merge_dim'],
        #     weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
        #     activation_fn=lrelu)

        x = slim.fully_connected(
            x,
            1,
            weights_regularizer=slim.l2_regularizer(subnet['l2-reg']),
            activation_fn=None)

        return x, h  # no explicit `sigmoid`


    def loss(self, x_s, y_s, x_t, y_t):
        '''  '''
        # def thresholding_add(x, v, minval=-1., maxval=1.):
        #     x_v = x + v
        #     x_v = tf.maximum(x_v, minval)
        #     x_v = tf.minimum(x_v, maxval)
        #     return x_v

        def circuit_loop(x, y):
            # v = decay * tf.random_normal(shape=tf.shape(x))

            z_mu, z_lv = self._encode(x, is_training=self.is_training)
            z = GaussianSampleLayer(z_mu, z_lv)

            x_logit, x_feature = self._discriminate(
                x, is_training=self.is_training)

            xh, xh_sig_logit = self._generate(z, y, is_training=self.is_training)

            zh_mu, zh_lv = self._encode(xh, is_training=self.is_training)

            xh_logit, xh_feature = self._discriminate(
                xh, is_training=self.is_training)

            return dict(
                z=z,
                z_mu=z_mu,
                z_lv=z_lv,
                # y_logit=y_logit,
                xh=xh,
                xh_sig_logit=xh_sig_logit,
                # xo=xo,
                x_logit=x_logit,
                x_feature=x_feature,
                zh_mu=zh_mu,
                zh_lv=zh_lv,
                # yh_logit=yh_logit,
                xh_logit=xh_logit,
                xh_feature=xh_feature,
                # xr=xr
                )

        s = circuit_loop(x_s, y_s)
        t = circuit_loop(x_t, y_t)
        s2t = circuit_loop(x_s, y_t)

        with tf.name_scope('loss'):
            def mean_sigmoid_cross_entropy_with_logits(logit, truth):
                '''
                truth: 0. or 1.
                '''
                return tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logit,
                        truth * tf.ones_like(logit)))

            loss = dict()

            # Parallel
            loss['reconst_t'] = \
                  tf.reduce_mean(t['x_logit']) \
                - tf.reduce_mean(t['xh_logit'])

            # Parallel
            loss['reconst_s'] = \
                  tf.reduce_mean(s['x_logit']) \
                - tf.reduce_mean(s['xh_logit'])

            # Non-parallel
            loss['conv_s2t'] = \
                  tf.reduce_mean(t['x_logit']) \
                - tf.reduce_mean(s2t['xh_logit'])

            # Non-parallel: s v. t
            loss['real_s_t'] = \
                  tf.reduce_mean(t['x_logit']) \
                - tf.reduce_mean(s['x_logit'])

            # That's why I only take the last term into consideration
            loss['WGAN'] = loss['conv_s2t']

            # VAE's Kullback-Leibler Divergence
            loss['KL(z)'] = \
                tf.reduce_mean(
                    GaussianKLD(
                        s['z_mu'], s['z_lv'],
                        tf.zeros_like(s['z_mu']), tf.zeros_like(s['z_lv']))) +\
                tf.reduce_mean(
                    GaussianKLD(
                        t['z_mu'], t['z_lv'],
                        tf.zeros_like(t['z_mu']), tf.zeros_like(t['z_lv'])))
            loss['KL(z)'] /= 2.0

            # VAE's Reconstruction Neg. Log-Likelihood (on the 'feature' space of Dx)
            loss['Dis'] = \
                tf.reduce_mean(
                    GaussianLogDensity(
                        slim.flatten(x_t),
                        slim.flatten(t['xh']),
                        tf.zeros_like(slim.flatten(x_t)))) +\
                tf.reduce_mean(
                    GaussianLogDensity(
                        slim.flatten(x_s),
                        slim.flatten(s['xh']),
                        tf.zeros_like(slim.flatten(x_s))))
            loss['Dis'] /= - 2.0

            # loss['Dis'] = tf.reduce_mean(
            #     tf.reduce_sum(
            #         tf.nn.sigmoid_cross_entropy_with_logits(
            #             slim.flatten(t['xh_sig_logit']),
            #             slim.flatten(x_t * .5 + .5)) +\
            #         tf.nn.sigmoid_cross_entropy_with_logits(
            #             slim.flatten(s['xh_sig_logit']),
            #             slim.flatten(x_s * .5 + .5)),
            #         -1))
            # # Note: Please remember to apply reduce_sum along the last dim
            # # (here, it's not 1 as is in D/G loss.)

            # loss['Dis'] = tf.reduce_mean(
            #     (s2t['xh_logit'] + s['xh_logit'] + t['xh_logit']) / 3.)

            # # Use 'feature' item instead of raw input because the former is 'flattened'
            # loss['Dis'] = tf.reduce_mean(
            #     tf.reduce_sum(tf.square(s['x_feature'] - s['xh_feature']), -1) +\
            #     tf.reduce_sum(tf.square(t['x_feature'] - t['xh_feature']), -1))

            # [TODO] Maybe I should normalize (divide by 2 or 3) the above costs.

            # For summaries
            with tf.name_scope('Summary'):
                # tf.summary.scalar('DKL_x', loss['KL(x)'])
                tf.summary.scalar('DKL_z', loss['KL(z)'])
                tf.summary.scalar('MMSE', loss['Dis'])

                # tf.summary.scalar('D_real', loss['D_real'])
                # tf.summary.scalar('G_fake', loss['G_fake'])

                tf.summary.scalar('WGAN', loss['WGAN'])
                tf.summary.scalar('WGAN-s', loss['reconst_s'])
                tf.summary.scalar('WGAN-t', loss['reconst_t'])
                tf.summary.scalar('WGAN-s2t', loss['conv_s2t'])
                tf.summary.scalar('WGAN-t-s', loss['real_s_t'])
                

                tf.summary.histogram('y', tf.concat(0, [y_t, y_s]))

                # tf.summary.histogram('yh_s', tf.argmax(s['yh_logit'], 1))
                # tf.summary.histogram('yh_t', tf.argmax(t['yh_logit'], 1))

                tf.summary.histogram('z', tf.concat(0, [s['z'], t['z']]))
                tf.summary.histogram('z_s', s['z'])
                tf.summary.histogram('z_t', t['z'])

                tf.summary.histogram('z_mu', tf.concat(0, [s['z_mu'], t['z_mu']]))
                tf.summary.histogram('z_mu_s', s['z_mu'])
                tf.summary.histogram('z_mu_t', t['z_mu'])

                tf.summary.histogram('z_lv', tf.concat(0, [s['z_lv'], t['z_lv']]))
                tf.summary.histogram('z_lv_s', s['z_lv'])
                tf.summary.histogram('z_lv_t', t['z_lv'])

                # tf.summary.histogram('D_t_t', tf.nn.sigmoid(x_t_from_t_logit))
                # tf.summary.histogram('D_t_s', tf.nn.sigmoid(x_t_from_s_logit))
                # tf.summary.histogram('D_t', tf.nn.sigmoid(x_t_logit))
                # tf.summary.histogram('D_s', tf.nn.sigmoid(x_s_logit))


                tf.summary.histogram('logit_t_from_t', t['xh_logit'])
                tf.summary.histogram('logit_t_from_s', s2t['xh_logit'])
                tf.summary.histogram('logit_t', t['x_logit'])
                # tf.summary.histogram('logit_s', s['x_logit'])

                # tf.summary.histogram(
                #     'logit_s_True_and_Fake)',
                #     tf.concat(0, [x_s_from_s_logit, x_s_logit]))
                tf.summary.histogram(
                    'logit_t_True_FromT_FromS',
                    tf.concat(0, [t['x_logit'], t['xh_logit'], s2t['xh_logit']]))
                tf.summary.histogram(
                    'logit_s_v_sh',
                    tf.concat(0, [s['x_logit'], s['xh_logit']]))
                tf.summary.histogram(
                    'logit_t_v_th',
                    tf.concat(0, [t['x_logit'], t['xh_logit']]))
                # # tf.image_summary("G", xh)

                # tf.summary.scalar('Ent_on_y', loss['y'])
                # tf.summary.scalar('Diff_zs_zt', loss['z'])
        return loss

    # def sample(self, z=128):
    #     ''' Generate fake samples given `z`
    #     if z is not given or is an `int`,
    #     this fcn generates (z=128) samples
    #     '''
    #     z = tf.random_uniform(
    #         shape=[z, self.arch['z_dim']],
    #         minval=-1.0,
    #         maxval=1.0,
    #         name='z_test')
    #     return self._generate(z, is_training=False)

    def encode(self, x):
        z_mu, z_lv = self._encode(x, is_training=False)
        return dict(mu=z_mu, log_var=z_lv)

    def decode(self, z, y, tanh=False):
        # if tanh:
        #     return self._generate(z, y, is_training=False)
        # else:
        #     return self._generate(z, y, is_training=False)
        xh, _ = self._generate(z, y, is_training=False)
        # tf.summary.image('xh', tf.transpose(xh, [2, 1, 0, 3]))
        # return self._filter(xh, is_training=False)
        return xh

    def discriminate(self, x):
        '''
        To estimate the EMD, we need D to assign a score per sample.
        *The batches can be of different size
        '''
        # s_true, _ = self._discriminate(x_true, is_training=False)
        # s_fake, _ = self._discriminate(x_fake, is_training=False)
        # return s_true, s_fake
        s, _ = self._discriminate(x, is_training=False)
        return s

    # def classify(self, x):
    #     return self._classify(tf.nn.softmax(x), is_training=False)

    # def interpolate(self, x1, x2, n):
    #     ''' Interpolation from the latent space '''
    #     x1 = tf.expand_dims(x1, 0)
    #     x2 = tf.expand_dims(x2, 0)
    #     z1, _ = self._encode(x1, is_training=False)
    #     z2, _ = self._encode(x2, is_training=False)
    #     a = tf.reshape(tf.linspace(0., 1., n), [n, 1])

    #     z1 = tf.matmul(1. - a, z1)
    #     z2 = tf.matmul(a, z2)
    #     z = tf.nn.tanh(tf.add(z1, z2))  # Gaussian-to-Uniform
    #     xh = self._generate(z, is_training=False)
    #     xh = tf.concat(0, [x1, xh, x2])
    #     return xh
