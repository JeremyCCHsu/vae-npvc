import json
import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from models import MLPcVAE
from vcc2016io import VCC2016TFRManager


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('logdir', 'logdir', 'log dir')
tf.app.flags.DEFINE_integer('target_id', 9, 'target id (SF1 = 1, TM3 = 9)')
tf.app.flags.DEFINE_string(
    'file_pattern', None, 'filename filter')

# speakers = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2', 'TF1', 'TF2', 'TM1', 'TM2', 'TM3']

def main():
    if args.logdir is None:
        raise ValueError('Please specify the logdir file')

    ckpt = get_checkpoint(args.logdir)
    
    if ckpt is None:
        raise ValueError('No checkpoints in {}'.format(args.logdir))

    with open(os.path.join(args.logdir, 'architecture.json')) as f:
        arch = json.load(f)

    reader = VCC2016TFRManager()
    features = reader.read_whole(args.file_pattern, num_epochs=1)
    x = features['frame']
    y = features['label']
    filename = features['filename']
    y_conv = y * 0 + args.target_id

    net = MLPcVAE(arch=arch, is_training=False)
    z = net.encode(x)
    xh = net.decode(z, y)
    x_conv = net.decode(z, y_conv)

    pre_train_saver = tf.train.Saver()
    def load_pretrain(sess):
        pre_train_saver.restore(sess, ckpt)
    sv = tf.train.Supervisor(init_fn=load_pretrain)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        gpu_options=gpu_options)
    with sv.managed_session(config=sess_config) as sess:
        for _ in range(reader.n_files):
            if sv.should_stop():
                break
            fetch_dict = {'x': x, 'xh': xh, 'x_conv': x_conv, 'f': filename}
            results = sess.run(fetch_dict)
            plot_spectra(results)


def get_checkpoint(logdir):
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        return ckpt.model_checkpoint_path
    else:
        print('No checkpoint found')
        return None


def plot_spectra(results):
    plt.figure(figsize=(10, 4))
    plt.imshow(
        np.concatenate(
            [np.flipud(results['x'].T),
             np.flipud(results['xh'].T),
             np.flipud(results['x_conv'].T)],
            0),
        aspect='auto',
        cmap='jet',
    )
    plt.colorbar()
    plt.title('Upper: Real input; Mid: Reconstrution; Lower: Conversion to target.')
    plt.savefig(
        os.path.join(
            args.logdir,
            '{}.png'.format(
                os.path.split(str(results['f'], 'utf-8'))[-1]
            )
        )
    )


if __name__ == '__main__':
    main()
