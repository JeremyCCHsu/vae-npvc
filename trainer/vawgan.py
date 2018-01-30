import logging
import os

import tensorflow as tf

from trainer.vae import VAETrainer
from util.wrapper import save


def lr_schedule(step, schedule):
    for s, lr in schedule:
        if step < s:
            return lr
    return 1e-7

class VAWGANTrainer(VAETrainer):
    def _optimize(self):
        hyperp = self.arch['training']
        global_step = tf.Variable(0, name='global_step')
        lr = tf.placeholder(dtype=tf.float32)  # can't add lr to summary

        optimizer_d = tf.train.RMSPropOptimizer(lr) #hyperp['lr'])
        optimizer_g = tf.train.RMSPropOptimizer(lr) #hyperp['lr'])

        trainables = tf.trainable_variables()
        g_vars = [v for v in trainables if 'Generator' in v.name]
        d_vars = [v for v in trainables if 'Discriminator' in v.name]
        e_vars = [v for v in trainables if 'Encoder' in v.name]

        # ======== Standard ========
        # r: a {0, R+} weighting 
        r = tf.placeholder(shape=[], dtype=tf.float32)
        k = tf.constant(hyperp['clamping'], shape=[])
        for term in ['reconst_t', 'reconst_s', 'conv_s2t', 'real_s_t']:
            self.loss[term] = self.loss[term] / k 

        obj_Dx = - self.loss['conv_s2t'] * k
        obj_Gx = r * self.loss['conv_s2t'] + self.loss['Dis']
        obj_Ez = self.loss['KL(z)'] + self.loss['Dis']

        opt_d = optimizer_d.minimize(obj_Dx, var_list=d_vars)
        opt_ds = [opt_d]

        logging.info('The following variables are clamped:')
        with tf.control_dependencies(opt_ds):
            with tf.name_scope('Clamping'):
                for v in d_vars:
                    v_clamped = tf.clip_by_value(v, -k, k)
                    clamping = tf.assign(v, v_clamped)
                    opt_ds.append(clamping)
                    logging.info(v.name)

        opt_g = optimizer_g.minimize(obj_Gx, var_list=g_vars, global_step=global_step)
        opt_e = optimizer_g.minimize(obj_Ez, var_list=e_vars)

        return dict(
            d=opt_ds,
            g=opt_g,
            gz=opt_e,
            lr=lr,
            gamma=r,
            global_step=global_step)


    def train(self, nIter, machine=None, summary_op=None):
        vae_saver = tf.train.Saver()
        sv = tf.train.Supervisor(
            logdir=self.dirs['logdir'],
            save_model_secs=300,
            global_step=self.opt['global_step'])

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        n_iter_per_epoch = 2 * int(1e5) // self.arch['training']['batch_size']  # TODO: should use #frames

        def print_log(info, results):
            msg = 'Epoch [{:3d}/{:3d}] '.format(info['ep'], info['nEpoch'])
            msg += '[{:4d}/{:4d}] '.format(info['it'], info['nIter'])
            msg += 'W: {:.2f} '.format(results['conv_s2t'])
            msg += 'DIS={:5.2f}, KLD={:5.2f}'.format(results['Dis'], results['KL(z)'])
            print('\r{}'.format(msg), end='', flush=True)
            logging.info(msg)


        hyperp = self.arch['training']
        info = {'nEpoch': hyperp['epoch_vae'], 'nIter': n_iter_per_epoch}
        fetches = {
            'conv_s2t': self.loss['conv_s2t'], 
            'Dis': self.loss['Dis'], 
            'KL(z)': self.loss['KL(z)'],
            'opt_g': self.opt['g'], 
            'opt_e': self.opt['gz'], 
            'step': self.opt['global_step'],
        }
        with sv.managed_session(config=sess_config) as sess:
            try: 
                update_G_E = [self.opt['g'], self.opt['gz']]

                for ep in range(hyperp['epoch_vae']):
                    lr = lr_schedule(ep, hyperp['lr_schedule'])
                    feed_dict = {self.opt['gamma']: 0., self.opt['lr']: lr}
                    for it in range(n_iter_per_epoch):
                        if it % 100 == 0:  # update print only every 100 iters
                            results = sess.run(fetches, feed_dict=feed_dict)
                            info.update({'ep': ep + 1, 'it': it + 1})
                            print_log(info, results)
                        else:
                            sess.run(update_G_E, feed_dict=feed_dict)

                save(vae_saver, sess, os.path.join(self.dirs['logdir'], 'VAE'), fetches['step'])  
                info.update({'nEpoch': hyperp['epoch_vawgan'] + hyperp['epoch_vae']})

                for ep in range(hyperp['epoch_vawgan']):
                    ep = ep + hyperp['epoch_vae']

                    lr = lr_schedule(ep, hyperp['lr_schedule'])
                    feed_dict = {self.opt['gamma']: hyperp['gamma'], self.opt['lr']: lr}

                    info.update({'ep': ep})
                    for it in range(n_iter_per_epoch):
                        if (ep == hyperp['epoch_vae'] and it < 25) or (it % 100 == 0):
                            nIterD = hyperp['n_unroll_intense']
                        else:
                            nIterD = hyperp['n_unroll']

                        for _ in range(nIterD):
                            sess.run(self.opt['d'], feed_dict=feed_dict)

                        if it % 100 != 0:
                            sess.run(update_G_E, feed_dict=feed_dict)
                        else:
                            results = sess.run(fetches, feed_dict=feed_dict)
                            info.update({'ep': ep + 1, 'it': it + 1})
                            print_log(info, results)
            except KeyboardInterrupt:
                print()
            finally:
                save(sv.saver, sess, self.dirs['logdir'], fetches['step'])
            print()

