import tensorflow as tf
from tensorflow.contrib import slim
from util.image import nchw_to_nhwc
from util.layers import (GaussianKLD, GaussianLogDensity, GaussianSampleLayer,
                         Layernorm, conv2d_nchw_layernorm, lrelu, selu,
                         selu_normal)
from models.wgan import GradientPenaltyWGAN


class FisherGAN(GradientPenaltyWGAN):
    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        with tf.name_scope('loss'):        
            z = self._generate_noise_with_shape(x)

            x_real = x
            x_fake = self._generate(z, y, is_training=self.is_training)

            y_all = None if y is None else tf.concat([y, y], 0)        
            x_all = tf.concat([x_real, x_fake], 0)

            c_ = self._discriminate(
                x=x_all,
                y=y_all,
                is_training=self.is_training
            )
            c_real, c_fake = tf.split(c_, 2)

            rho = self.arch['training']['rho']

            E_real = tf.reduce_mean(c_real)
            E_fake = tf.reduce_mean(c_fake)
            E_dist = E_real - E_fake

            E_real2 = tf.reduce_mean(tf.square(c_real))
            E_fake2 = tf.reduce_mean(tf.square(c_fake))
            Omega = E_real2 + E_fake2

            constraint = 1. - 0.5 * Omega
            alm = rho / 2. * tf.square(constraint)

            loss = dict()
            lam = tf.Variable(0.0, name='lambda')
            loss['l_G'] = - E_fake   # equiv. to E_dist because E_real is uncorrelated to G
            loss['l_D'] = - (E_dist + lam * constraint - alm)  # critic: max_p min_l L
            loss['IPM'] = E_dist / tf.sqrt(0.5 * Omega)

        # For summaries
        tf.summary.scalar('E_real', E_real)
        tf.summary.scalar('E_fake', E_fake)
        tf.summary.scalar('l_D', loss['l_D'])
        tf.summary.scalar('lambda', lam)
        tf.summary.scalar('IPM', loss['IPM'])
        tf.summary.scalar('E_dist', E_dist)
        tf.summary.histogram('c_real', c_real)
        tf.summary.histogram('c_fake', c_fake)
        tf.summary.histogram('x_real', slim.flatten(x_real))
        tf.summary.histogram('x_fake', slim.flatten(x_fake))
        tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
        tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss

        
class BiFisherGAN(FisherGAN):
    ''' Conditional on `y` '''
    def __init__(self, arch, is_training=False):
        super(BiFisherGAN, self).__init__(arch, is_training)
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

        # Radford: [do] not applying batchnorm to the discriminator input layer
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu, name='Conv2d-{}'.format(i))

        x = slim.flatten(x)
        x = tf.concat([x, z], axis=1)

        if y is not None:
            y_emb = self.make_y_feature_map(y, self.y_emb)
            x = tf.concat([x, y_emb], 1)

        x = tf.layers.dense(x, 512)
        x = Layernorm(x, [1], 'layernorm-last1')
        x = lrelu(x)

        x = tf.layers.dense(x, 512)
        x = Layernorm(x, [1], 'layernorm-last2')
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

        x_real = x
        x_fake = self._generate(z, y, is_training=self.is_training)
        
        x_all = tf.concat([x_real, x_fake], 0)
        y_all = tf.concat([y, y], 0) if y is not None else None
        z_all = tf.concat([z_real, z], 0)
        
        c_ = self._discriminate(
            x=x_all,
            y=y_all,
            z=z_all,
            is_training=self.is_training
        )
        c_real, c_fake = tf.split(c_, 2)

        with tf.name_scope('loss'):
            rho = self.arch['training']['rho']

            E_real = tf.reduce_mean(c_real)
            E_fake = tf.reduce_mean(c_fake)
            E_dist = E_real - E_fake

            E_real2 = tf.reduce_mean(tf.square(c_real))
            E_fake2 = tf.reduce_mean(tf.square(c_fake))
            Omega = E_real2 + E_fake2

            constraint = 1. - 0.5 * Omega
            alm = rho / 2. * tf.square(constraint)

            loss = dict()
            lam = tf.Variable(0.0, name='lambda')
            # L = E_dist + lam * constraint - alm
            loss['lam'] = lam
            loss['l_G'] = - E_fake   # equiv. to E_dist because E_real is uncorrelated to G
            loss['l_D'] = - (E_dist + lam * constraint - alm)  # critic: max_p min_l L
            loss['IPM'] = E_dist / tf.sqrt(0.5 * Omega)

            # For summaries
            tf.summary.scalar('E_real', E_real)
            tf.summary.scalar('E_fake', E_fake)
            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('lambda', loss['lam'])
            tf.summary.scalar('IPM', loss['IPM'])
            tf.summary.scalar('E_dist', E_dist)
            tf.summary.scalar('Omega', Omega)
            tf.summary.histogram('c_real', c_real)
            tf.summary.histogram('c_fake', c_fake)
            tf.summary.histogram('z_enc', z_real)
            tf.summary.histogram('z', z)
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake))

            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss



class CycleFisherGAN(FisherGAN):
    ''' Conditional on `y` '''
    def __init__(self, arch, is_training=False):
        super(CycleFisherGAN, self).__init__(arch, is_training)
        self._encode = tf.make_template('Encoder', self._encoder)
        self._Dz = tf.make_template('Dz', self._D_MLP)
    
    def _D_MLP(self, x):
        net = self.arch['discriminator_z']
        for i, o in enumerate(net['output']):
            activation = tf.identity if i == len(net['output']) - 1 else lrelu
            x = tf.layers.dense(x, o)
            x = Layernorm(x, [1], 'layernorm-last-{}'.format(i))
            x = activation(x)
        return x        

    def _encoder(self, x, is_training):
        # Radford: [do] not applying batchnorm to the discriminator input layer
        net = self.arch['discriminator']
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu, name='Conv2d-{}'.format(i))
        x = slim.flatten(x)
        x = tf.layers.dense(x, self.arch['z_dim'])
        return x

    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        z = self._generate_noise_with_shape(x)
        z_enc = self._encode(x, is_training=self.is_training)
        z_enc += z / 10.   # SNR = 20 dB

        z_all = tf.concat([z_enc, z], 0)
        y_all = tf.concat([y, y], 0) if y is not None else None       
        x_ = self._generate(z_all, y_all, is_training=self.is_training)

        x_r, x_fake = tf.split(x_, 2)
        
        x_real = x
        
        x_all = tf.concat([x_real, x_fake], 0)
        
        c_ = self._discriminate(
            x=x_all,
            y=y_all,
            is_training=self.is_training
        )
        c_real, c_fake = tf.split(c_, 2)

        cz_ = self._Dz(z_all)
        cz_fake, cz_real = tf.split(cz_, 2)

        with tf.name_scope('loss'):
            rho = self.arch['training']['rho']

            E_real = tf.reduce_mean(c_real)
            E_fake = tf.reduce_mean(c_fake)
            E_dist = E_real - E_fake

            E_real2 = tf.reduce_mean(tf.square(c_real))
            E_fake2 = tf.reduce_mean(tf.square(c_fake))
            Omega = E_real2 + E_fake2

            constraint = 1. - 0.5 * Omega
            alm = rho / 2. * tf.square(constraint)


            e_real = tf.reduce_mean(cz_real)
            e_fake = tf.reduce_mean(cz_fake)
            e_dist = e_real - e_fake

            e_real2 = tf.reduce_mean(tf.square(cz_real))
            e_fake2 = tf.reduce_mean(tf.square(cz_fake))
            omega = e_real2 + e_fake2

            constraintz = 1. - .5 * omega
            almz = rho / 2. * tf.square(constraintz)


            l_x = tf.reduce_mean(tf.reduce_sum(tf.abs(x - x_r), [1, 2, 3]))

            loss = dict()
            lam = tf.Variable(0.0, name='lambda')
            # L = E_dist + lam * constraint - alm
            loss['lam'] = lam
            loss['l_G'] = - E_fake + l_x  # equiv. to E_dist because E_real is uncorrelated to G
            loss['l_D'] = - (E_dist + lam * constraint - alm)  # critic: max_p min_l L
            loss['IPM'] = E_dist / tf.sqrt(0.5 * Omega)


            lamz = tf.Variable(0.0, name='lambdz')
            loss['l_E'] = - e_fake + l_x
            loss['l_Dz'] = - (e_dist + lamz * constraintz - almz)


            # For summaries
            tf.summary.scalar('l_E', loss['l_E'])
            tf.summary.scalar('l_Dz', loss['l_Dz'])

            tf.summary.scalar('E_real', E_real)
            tf.summary.scalar('E_fake', E_fake)
            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('lambda', loss['lam'])
            tf.summary.scalar('IPM', loss['IPM'])
            tf.summary.scalar('E_dist', E_dist)
            tf.summary.scalar('Omega', Omega)
            tf.summary.histogram('c_real', c_real)
            tf.summary.histogram('c_fake', c_fake)
            tf.summary.histogram('z_enc', z_enc)
            tf.summary.histogram('z', z)
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake))

            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss


class VAEFisherGAN(FisherGAN):
    ''' Conditional on `y` '''
    def __init__(self, arch, is_training=False):
        super(VAEFisherGAN, self).__init__(arch, is_training)
        self._encode = tf.make_template('Encoder', self._encoder)

    def _encoder(self, x, is_training):
        # Radford: [do] not applying batchnorm to the discriminator input layer
        net = self.arch['discriminator']
        for i, (o, k, s) in enumerate(zip(net['output'], net['kernel'], net['stride'])):
            x = conv2d_nchw_layernorm(
                x, o, k, s, lrelu, name='Conv2d-{}'.format(i))
        x = slim.flatten(x)
        z_mu = tf.layers.dense(x, self.arch['z_dim'])
        z_lv = tf.layers.dense(x, self.arch['z_dim'])
        return z_mu, z_lv

    # TODO: I should fix `y_emb`

    def loss(self, x, y):
        '''
        Return:
            `loss`: a `dict` for the trainer (keys must match)
        '''
        z = self._generate_noise_with_shape(x)
        z_mu, z_lv = self._encode(x, is_training=self.is_training)
        z_enc = GaussianSampleLayer(z_mu, z_lv)

        z_all = tf.concat([z_enc, z], 0)
        y_all = tf.concat([y, y], 0) if y is not None else None       
        x_ = self._generate(z_all, y_all, is_training=self.is_training)

        x_r, x_fake = tf.split(x_, 2)
        
        x_real = x
        
        x_all = tf.concat([x_real, x_fake], 0)
        
        c_ = self._discriminate(
            x=x_all,
            y=y_all,
            is_training=self.is_training
        )
        c_real, c_fake = tf.split(c_, 2)

        with tf.name_scope('loss'):

            E_real = tf.reduce_mean(c_real)
            E_fake = tf.reduce_mean(c_fake)
            E_dist = E_real - E_fake

            E_real2 = tf.reduce_mean(tf.square(c_real))
            E_fake2 = tf.reduce_mean(tf.square(c_fake))
            Omega = E_real2 + E_fake2

            rho = self.arch['training']['rho']           
            constraint = 1. - 0.5 * Omega
            alm = rho / 2. * tf.square(constraint)

            # VAE loss
            l_x = - tf.reduce_mean(GaussianLogDensity(x, x_r, tf.zeros_like(x_r)))

            l_KLD = tf.reduce_mean(
                GaussianKLD(
                    z_mu, z_lv,
                    tf.zeros_like(z_mu), tf.zeros_like(z_lv)
                )
            )

            loss = dict()
            lam = tf.Variable(0.0, name='lambda')
            loss['lam'] = lam
            loss['l_G'] = - E_fake + l_x + l_KLD # equiv. to E_dist because E_real is uncorrelated to G
            loss['l_D'] = - (E_dist + lam * constraint - alm)  # critic: max_p min_l L
            loss['IPM'] = E_dist / tf.sqrt(0.5 * Omega)

            # For summaries
            tf.summary.scalar('x_GaussLogProb', l_x)
            tf.summary.scalar('KLD_z', l_KLD)
            
            tf.summary.scalar('E_real', E_real)
            tf.summary.scalar('E_fake', E_fake)
            tf.summary.scalar('l_D', loss['l_D'])
            tf.summary.scalar('lambda', loss['lam'])
            tf.summary.scalar('IPM', loss['IPM'])
            tf.summary.scalar('E_dist', E_dist)
            tf.summary.scalar('Omega', Omega)
            tf.summary.histogram('c_real', c_real)
            tf.summary.histogram('c_fake', c_fake)
            tf.summary.histogram('x_real', slim.flatten(x_real))
            tf.summary.histogram('x_fake', slim.flatten(x_fake))

            tf.summary.image('1_real', nchw_to_nhwc(x_real), max_outputs=4)
            tf.summary.image('0_fake', nchw_to_nhwc(x_fake), max_outputs=4)

        return loss
