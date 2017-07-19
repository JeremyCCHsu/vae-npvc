import tensorflow as tf
import numpy as np
import logging, os
from util.image import make_png_thumbnail, make_png_jet_thumbnail

def debug(labels, var_list):
    for name, vs in zip(labels, var_list):
        print(name)
        for v in vs:
            print(v.name)
        print()

class GANTrainer(object):
    def __init__(self, loss, arch, args, dirs):
        self.loss = loss
        self.arch = arch
        self.args = args
        self.dirs = dirs
        self.opt = self._optimize()
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(dirs['logdir'], 'training.log')
        )

    def _optimize(self):
        '''
        NOTE: The author said that there was no need for 100 d_iter per 100 iters. 
              https://github.com/igul222/improved_wgan_training/issues/3
        '''
        global_step = tf.Variable(0, name='global_step')
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']

        optimizer = tf.train.AdamOptimizer(lr, b1, b2)

        trainables = tf.trainable_variables()
        g_vars = [v for v in trainables if 'Generator' in v.name or 'y_emb' in v.name]
        d_vars = [v for v in trainables if 'Discriminator' in v.name]

        # # Debug ===============
        # debug(['Generator', 'Discriminator'], [g_vars, d_vars])
        # # ============================

        with tf.name_scope('Update'):        
            opt_g = optimizer.minimize(self.loss['l_G'], var_list=g_vars, global_step=global_step)
            opt_d = optimizer.minimize(self.loss['l_D'], var_list=d_vars)
        return {
            'd': opt_d,
            'g': opt_g,
            'global_step': global_step
        }


    def _validate(self, machine, n=10):
        N = n * n

        z = np.random.uniform(-np.pi, np.pi, size=[n, self.arch['z_dim']])
        z = np.cos(z)
        z = np.concatenate([z] * n, axis=1)
        z = np.reshape(z, [N, -1]).astype(np.float32)  # consecutive rows
        # y = np.random.randint(1, arch['y_dim'], size=[N, 2])
        y = np.asarray(
            [[5,   0,  0 ],  # 5
            [9,   0,  0 ],   # 9
            [12,  0,  0 ],   # 2
            [17,  0,  0 ],   # 7
            [19,  0,  0 ],
            [161, 0,  0 ],
            [170, 0,  0 ],
            [170, 16, 0 ],
            [161, 9,  4 ],
            [19,  24, 50]],
            dtype=np.int64)
        y = np.concatenate([y] * n, axis=0)

        Z = tf.constant(z)
        Y = tf.constant(y)
        Xh = machine.generate(Z, Y) # 100, 64, 64, 3
        Xh = make_png_thumbnail(Xh, n)
        return Xh


    def _refresh_status(self, sess):
        fetches = {
            "l_D": self.loss['l_D'],
            "l_G": self.loss['l_G'], 
            "step": self.opt['global_step'],
        }
        result = sess.run(
            fetches=fetches,
            # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            # run_metadata=run_metadata,
        )

        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open(os.path.join(dirs['logdir'], 'timeline.ctf.json'), 'w') as fp:
        #     fp.write(trace.generate_chrome_trace_format())

        # Message
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'l_D={:.3e} '.format(result['l_D'])
        msg += 'l_G={:.3e} '.format(result['l_G'])        
        print('\r{}'.format(msg), end='', flush=True)
        logging.info(msg)


    def train(self, nIter, machine=None, summary_op=None):
        Xh = self._validate(machine=machine, n=10)

        run_metadata = tf.RunMetadata()

        # summary_op = tf.summary.merge_all()

        sv = tf.train.Supervisor(
            logdir=self.dirs['logdir'],
            # summary_writer=summary_writer,
            # summary_op=None,
            # is_chief=True,
            # save_model_secs=600,
            global_step=self.opt['global_step'])


        # sess_config = configure_gpu_settings(args.gpu_cfg)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        with sv.managed_session(config=sess_config) as sess:
            sv.loop(60, self._refresh_status, (sess,))
            for step in range(self.arch['training']['max_iter']):
                if sv.should_stop():
                    break

                # main loop
                for _ in range(self.arch['training']['nIterD']):
                    sess.run(self.opt['d'])
                sess.run(self.opt['g'])

                # output img                
                if step % 1000 == 0:
                    xh = sess.run(Xh)
                    with tf.gfile.GFile(
                        os.path.join(
                            self.dirs['logdir'],
                            'img-anime-{:03d}k.png'.format(step // 1000),
                        ),
                        mode='wb',
                    ) as fp:
                        fp.write(xh)




class FisherGANTrainer(GANTrainer):
    def _optimize(self):
        '''
        NOTE: The author said that there was no need for 100 d_iter per 100 iters. 
              https://github.com/igul222/improved_wgan_training/issues/3
        '''
        global_step = tf.Variable(0, name='global_step')
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']
        rho = self.arch['training']['rho']

        optimizer = tf.train.AdamOptimizer(lr, b1, b2)
        optimizer_l = tf.train.GradientDescentOptimizer(rho)

        trainables = tf.trainable_variables()
        g_vars = [v for v in trainables if 'Generator' in v.name or 'y_emb' in v.name]
        d_vars = [v for v in trainables if 'Discriminator' in v.name]
        l_vars = [v for v in trainables if 'lambda' in v.name]
        
        # # Debug ===============
        # debug(['G', 'D', 'lambda'], [g_vars, d_vars, l_vars])
        # # ============================

        with tf.name_scope('Update'):        
            opt_g = optimizer.minimize(self.loss['l_G'], var_list=g_vars, global_step=global_step)
            opt_l = optimizer_l.minimize(- self.loss['l_D'], var_list=l_vars)
            with tf.control_dependencies([opt_l]):
                opt_d = optimizer.minimize(self.loss['l_D'], var_list=d_vars)            
        return {
            'd': opt_d,
            'g': opt_g,
            'l': opt_l,
            'global_step': global_step
        }

    def _validate(self, machine, n=10):
        N = n * n

        # same row same z
        z = tf.random_normal(shape=[n, self.arch['z_dim']])
        z = tf.tile(z, [1, n])
        z = tf.reshape(z, [N, -1])
        z = tf.Variable(z, trainable=False, dtype=tf.float32)       

        # same column same y 
        y = tf.range(0, 10, 1, dtype=tf.int64)
        y = tf.reshape(y, [-1, 1])
        y = tf.tile(y, [n, 1])

        Xh = machine.generate(z, y) # 100, 64, 64, 3
        # Xh = gray2jet(Xh)
        # Xh = make_png_thumbnail(Xh, n)
        Xh = make_png_jet_thumbnail(Xh, n)
        return Xh


class FisherGANTrainerHw3(FisherGANTrainer):
    def _validate(self, machine, n=10):
        N = n * n
        z = np.random.normal(0., 1., size=[n, self.arch['z_dim']])
        z = np.concatenate([z] * n, axis=1)
        z = np.reshape(z, [N, -1]).astype(np.float32)  # consecutive rows
        y = np.asarray(
            [[5,   0,  0 ],
             [9,   0,  0 ],
             [12,  0,  0 ],
             [17,  0,  0 ],
             [19,  0,  0 ],
             [161, 0,  0 ],
             [170, 0,  0 ],
             [170, 16, 0 ],
             [161, 9,  4 ],
             [19,  24, 50]],
            dtype=np.int64)
        y = np.concatenate([y] * n, axis=0)
        Z = tf.constant(z)
        Y = tf.constant(y)
        Xh = machine.generate(Z, Y) # 100, 64, 64, 3
        Xh = make_png_thumbnail(Xh, n)
        return Xh



class BiFisherGANTrainer(FisherGANTrainer):
    def _optimize(self):
        '''
        NOTE: The author said that there was no need for 100 d_iter per 100 iters. 
              https://github.com/igul222/improved_wgan_training/issues/3
        '''
        global_step = tf.Variable(0, name='global_step')
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']
        rho = self.arch['training']['rho']

        optimizer = tf.train.AdamOptimizer(lr, b1, b2)
        optimizer_l = tf.train.GradientDescentOptimizer(rho)

        trainables = tf.trainable_variables()
        g_vars = [v for v in trainables if 'Generator' in v.name or 'y_emb' in v.name]
        d_vars = [v for v in trainables if 'Discriminator' in v.name]
        l_vars = [v for v in trainables if 'lambda' in v.name]
        e_vars = [v for v in trainables if 'Encoder' in v.name]
        
        # # Debug ===============
        # for name, vs in zip(
        #     ['Generator', 'Discriminator', 'Encoder', 'lambda'],
        #     [g_vars, d_vars, e_vars, l_vars]):
        #     print(name)
        #     for v in vs:
        #         print(v.name)
        #     print()
        # # ============================

        with tf.name_scope('Update'):
            opt_e = optimizer.minimize(self.loss['l_G'], var_list=e_vars)
            with tf.control_dependencies([opt_e]):
                opt_g = optimizer.minimize(self.loss['l_G'], var_list=g_vars, global_step=global_step)
            
            opt_l = optimizer_l.minimize(- self.loss['l_D'], var_list=l_vars)  # opposite sign to D
            with tf.control_dependencies([opt_l]):
                opt_d = optimizer.minimize(self.loss['l_D'], var_list=d_vars)            
        return {
            'd': opt_d,
            'g': opt_g,
            'l': opt_l,
            'global_step': global_step
        }

class CycleFisherGANTrainer(FisherGANTrainer):
    def _optimize(self):
        '''
        NOTE: The author said that there was no need for 100 d_iter per 100 iters. 
              https://github.com/igul222/improved_wgan_training/issues/3
        '''
        global_step = tf.Variable(0, name='global_step')
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']
        rho = self.arch['training']['rho']

        optimizer = tf.train.AdamOptimizer(lr, b1, b2)
        optimizer_l = tf.train.GradientDescentOptimizer(rho)

        trainables = tf.trainable_variables()
        g_vars = [v for v in trainables if 'Generator' in v.name or 'y_emb' in v.name]
        
        d_vars = [v for v in trainables if 'Discriminator' in v.name]
        l_vars = [v for v in trainables if 'lambda' in v.name]
        
        e_vars = [v for v in trainables if 'Encoder' in v.name]

        z_vars = [v for v in trainables if 'Dz']       
        j_vars = [v for v in trainables if 'lambdz' in v.name]
        # # Debug ===============
        # # ============================

        with tf.name_scope('Update'):           
            opt_j = optimizer_l.minimize(- self.loss['l_Dz'], var_list=j_vars)
            opt_z = optimizer.minimize(self.loss['l_Dz'], var_list=z_vars)

            opt_e = optimizer.minimize(self.loss['l_E'], var_list=e_vars)
            with tf.control_dependencies([opt_e, opt_j, opt_z]):
                opt_g = optimizer.minimize(self.loss['l_G'], var_list=g_vars, global_step=global_step)
            
            opt_l = optimizer_l.minimize(- self.loss['l_D'], var_list=l_vars)  # opposite sign to D
            with tf.control_dependencies([opt_l]):
                opt_d = optimizer.minimize(self.loss['l_D'], var_list=d_vars)            
        return {
            'd': opt_d,
            'g': opt_g,
            'l': opt_l,
            'global_step': global_step
        }


# class BEGANTrainer(GANTrainer):
#     def _optimize(self):  #loss, args, arch, net=None):
#         '''
#         [TODO]
#         Although most of the trainer structures are the same,
#         I think we have to use different training scripts for VAE- and DC-GAN
#         (but do we have to have two different classes of VAE- and DC-?)
#         '''
#         global_step = tf.Variable(0, name='global_step')
#         lr = self.arch['training']['lr']
#         # optimizer_d = tf.train.AdamOptimizer(lr)
#         optimizer_g = tf.train.AdamOptimizer(lr)

#         trainables = tf.trainable_variables()
#         g_vars = [v for v in trainables if 'Generator' in v.name or 'y_emb' in v.name]
#         d_vars = [v for v in trainables if 'Critic' in v.name]

#         # Debug ===============
#         for name, vs in zip(
#             ['Generator', 'Critic'],
#             [g_vars, d_vars]):
#             print(name)
#             for v in vs:
#                 print(v.name)
#             print()
#         # ============================

#         with tf.name_scope('Update'):        
#             opt_g = optimizer_g.minimize(self.loss['G'], var_list=g_vars)
#             opt_d = optimizer_g.minimize(self.loss['D'], global_step=global_step, var_list=d_vars)
#             with tf.control_dependencies([opt_d, opt_g]):
#                 k_delta = self.arch['training']['lambda'] * self.loss['k_delta']
#                 opt_k = tf.assign(
#                     net.k_t, tf.clip_by_value(net.k_t + k_delta, 0, 1))

#         # logging.info('The following variables are clamped:')

#         return {
#             'd': opt_d,
#             'g': opt_g,
#             'k': opt_k,
#             'global_step': global_step
#         }

class VAETrainer(GANTrainer):
    def _optimize(self):
        '''
        NOTE: The author said that there was no need for 100 d_iter per 100 iters. 
              https://github.com/igul222/improved_wgan_training/issues/3
        '''
        global_step = tf.Variable(0, name='global_step')
        lr = self.arch['training']['lr']
        b1 = self.arch['training']['beta1']
        b2 = self.arch['training']['beta2']

        optimizer = tf.train.AdamOptimizer(lr, b1, b2)

        trainables = tf.trainable_variables()
        g_vars = trainables
        # g_vars = [v for v in trainables if 'Generator' in v.name or 'y_emb' in v.name]

        with tf.name_scope('Update'):        
            opt_g = optimizer.minimize(self.loss['G'], var_list=g_vars, global_step=global_step)
        return {
            'g': opt_g,
            'global_step': global_step
        }


    def _refresh_status(self, sess):
        fetches = {
            "D_KL": self.loss['D_KL'],
            "logP": self.loss['logP'],
            "step": self.opt['global_step'],
        }
        result = sess.run(
            fetches=fetches,
            # options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
            # run_metadata=run_metadata,
        )

        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # with open(os.path.join(dirs['logdir'], 'timeline.ctf.json'), 'w') as fp:
        #     fp.write(trace.generate_chrome_trace_format())

        # Message
        msg = 'Iter {:05d}: '.format(result['step'])
        msg += 'log P(x|z, y) = {:.3e} '.format(result['logP'])
        msg += 'D_KL(z) = {:.3e} '.format(result['D_KL'])
        print('\r{}'.format(msg), end='', flush=True)
        logging.info(msg)


    def _validate(self, machine, n=10):
        N = n * n
        
        # same row same z
        z = tf.random_normal(shape=[n, self.arch['z_dim']])
        z = tf.tile(z, [1, n])
        z = tf.reshape(z, [N, -1])
        z = tf.Variable(z, trainable=False, dtype=tf.float32)       

        # same column same y 
        y = tf.range(0, 10, 1, dtype=tf.int64)
        y = tf.reshape(y, [-1,])
        y = tf.tile(y, [n,])

        Xh = machine.generate(z, y) # 100, 64, 64, 3
        Xh = make_png_thumbnail(Xh, n)
        return Xh

    def train(self, nIter, machine=None, summary_op=None):
        Xh = self._validate(machine=machine, n=10)

        run_metadata = tf.RunMetadata()

        sv = tf.train.Supervisor(
            logdir=self.dirs['logdir'],
            # summary_writer=summary_writer,
            # summary_op=None,
            # is_chief=True,
            # save_model_secs=600,
            global_step=self.opt['global_step'])


        # sess_config = configure_gpu_settings(args.gpu_cfg)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        with sv.managed_session(config=sess_config) as sess:
            sv.loop(60, self._refresh_status, (sess,))
            for step in range(self.arch['training']['max_iter']):
                if sv.should_stop():
                    break

                # main loop
                sess.run(self.opt['g'])

                # output img                
                if step % 1000 == 0:
                    xh = sess.run(Xh)
                    with tf.gfile.GFile(
                        os.path.join(
                            self.dirs['logdir'],
                            'img-anime-{:03d}k.png'.format(step // 1000),
                        ),
                        mode='wb',
                    ) as fp:
                        fp.write(xh)
