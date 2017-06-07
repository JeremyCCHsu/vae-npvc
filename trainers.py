import logging
import os

import tensorflow as tf


class VAETrainer(object):
    ''' VAE trainer using VAW-GAN model '''

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
        global_step = tf.Variable(0, name='global_step')  # NOTE: MUST!
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.arch['training']['lr'],
            beta1=self.arch['training']['beta1'],
            beta2=self.arch['training']['beta2'],
        )
        obj = self.loss['L_x'] + self.loss['L_z']
        opt = optimizer.minimize(obj, global_step=global_step)
        return {
            'g': opt,
            'global_step': global_step
        }

    def train(self, nIter, summary_op=None):
        ''' main training loop '''
        run_metadata = tf.RunMetadata()

        sv = tf.train.Supervisor(
            logdir=self.dirs['logdir'],
            save_model_secs=60 * 2.5,
            global_step=self.opt['global_step'])
        # sess_config = configure_gpu_settings(args.gpu_cfg)
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options)
        with sv.managed_session(config=sess_config) as sess:
            for step in range(nIter):
                if sv.should_stop():
                    break
                sess.run(self.opt['g'])
                print('\rIter {:06d}'.format(step), end='', flush=True)

                if step % self.args.summary_freq == 0:
                    fetch_dict = {'l_x': self.loss['L_x'], 'l_z': self.loss['L_z']}
                    results = sess.run(fetch_dict)
                
                    msg = '\rIter {:06d}: E[ln p(x|z, y)]={:.4f}, D_KL={:.4f}'.format(
                        step,
                        results['l_x'],
                        results['l_z'],
                    )  
                    print(msg)
