import logging
import os

import numpy as np
import tensorflow as tf
# from util.image import make_png_jet_thumbnail, make_png_thumbnail
from trainer.gan import GANTrainer

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

        g_vars = tf.trainable_variables()

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


    # def _validate(self, machine, n=10):
    #     N = n * n

    #     # same row same z
    #     z = tf.random_normal(shape=[n, self.arch['z_dim']])
    #     z = tf.tile(z, [1, n])
    #     z = tf.reshape(z, [N, -1])
    #     z = tf.Variable(z, trainable=False, dtype=tf.float32)

    #     # same column same y
    #     y = tf.range(0, 10, 1, dtype=tf.int64)
    #     y = tf.reshape(y, [-1,])
    #     y = tf.tile(y, [n,])

    #     Xh = machine.generate(z, y) # 100, 64, 64, 3
    #     Xh = make_png_thumbnail(Xh, n)
    #     return Xh

    def train(self, nIter, machine=None, summary_op=None):
        # Xh = self._validate(machine=machine, n=10)

        run_metadata = tf.RunMetadata()

        sv = tf.train.Supervisor(
            logdir=self.dirs['logdir'],
            # summary_writer=summary_writer,
            # summary_op=None,
            # is_chief=True,
            save_model_secs=300,
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

                # # output img
                # if step % 1000 == 0:
                #     xh = sess.run(Xh)
                #     with tf.gfile.GFile(
                #         os.path.join(
                #             self.dirs['logdir'],
                #             'img-anime-{:03d}k.png'.format(step // 1000),
                #         ),
                #         mode='wb',
                #     ) as fp:
                #         fp.write(xh)



class VAWGANTrainer(GANTrainer):
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
        e_vars = [v for v in trainables if 'Encoder' in v.name]

        # # Debug ===============
        # debug(['Generator', 'Discriminator'], [g_vars, d_vars])
        # # ============================

        with tf.name_scope('Update'):
            opt_d = optimizer.minimize(self.loss['l_D'], var_list=d_vars)
            opt_e = optimizer.minimize(self.loss['l_E'], var_list=e_vars)
            with tf.control_dependencies([opt_e]):
                opt_g = optimizer.minimize(self.loss['l_G'], var_list=g_vars, global_step=global_step)
        return {
            'd': opt_d,
            'g': opt_g,
            'e': opt_e,
            'global_step': global_step
        }

    def train(self, nIter, machine=None, summary_op=None):
        # Xh = self._validate(machine=machine, n=10)

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

                # # output img
                # if step % 1000 == 0:
                #     xh = sess.run(Xh)
                #     with tf.gfile.GFile(
                #         os.path.join(
                #             self.dirs['logdir'],
                #             'img-anime-{:03d}k.png'.format(step // 1000),
                #         ),
                #         mode='wb',
                #     ) as fp:
                #         fp.write(xh)

    def _refresh_status(self, sess):
        fetches = {
            "D_KL": self.loss['D_KL'],
            "logP": self.loss['logP'],
            "W_dist": self.loss['W_dist'],
            "gp": self.loss['gp'],
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
        msg += 'W_dist = {:.4e} '.format(result['W_dist'])
        msg += 'log P(x|z, y) = {:.4e} '.format(result['logP'])
        msg += 'D_KL(z) = {:.4e} '.format(result['D_KL'])
        msg += 'GP = {:.4e} '.format(result['gp'])
        print('\r{}'.format(msg), end='', flush=True)
        logging.info(msg)