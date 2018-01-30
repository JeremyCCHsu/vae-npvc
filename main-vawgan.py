import json
import os

from importlib import import_module

import numpy as np
import tensorflow as tf

from analyzer import Tanhize, read
from util.wrapper import validate_log_dirs

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'corpus_name', 'vcc2016', 'Corpus name')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_string('gpu_cfg', None, 'GPU configuration')
tf.app.flags.DEFINE_integer('summary_freq', 1000, 'Update summary')
tf.app.flags.DEFINE_string(
    'ckpt', None, 'specify the ckpt in restore_from (if there are multiple ckpts)')  # TODO
tf.app.flags.DEFINE_string(
    'architecture', 'architecture-vawgan-vcc2016.json', 'network architecture')

tf.app.flags.DEFINE_string('model_module', 'model.vawgan', 'Model module')
tf.app.flags.DEFINE_string('model', 'VAWGAN', 'Model: ConvVAE, VAWGAN')

tf.app.flags.DEFINE_string('trainer_module', 'trainer.vawgan', 'Trainer module')
tf.app.flags.DEFINE_string('trainer', 'VAWGANTrainer', 'Trainer: VAETrainer, VAWGANTrainer')


def main(unused_args=None):
    ''' NOTE: The input is rescaled to [-1, 1] '''
    module = import_module(args.model_module, package=None)
    MODEL = getattr(module, args.model)

    module = import_module(args.trainer_module, package=None)
    TRAINER = getattr(module, args.trainer)


    dirs = validate_log_dirs(args)
    tf.gfile.MakeDirs(dirs['logdir'])

    with open(args.architecture) as f:
        arch = json.load(f)

    with open(os.path.join(dirs['logdir'], args.architecture), 'w') as f:
        json.dump(arch, f, indent=4)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/{}_xmax.npf'.format(args.corpus_name)),
        xmin=np.fromfile('./etc/{}_xmin.npf'.format(args.corpus_name)),
    )

    x_s, y_s = read(
        file_pattern=arch['training']['src_dir'],
        batch_size=arch['training']['batch_size'],
        capacity=2048,
        min_after_dequeue=1024,
        normalizer=normalizer,
        data_format='NHWC',
    )

    x_t, y_t = read(
        file_pattern=arch['training']['trg_dir'],
        batch_size=arch['training']['batch_size'],
        capacity=2048,
        min_after_dequeue=1024,
        normalizer=normalizer,
        data_format='NHWC',
    )

    machine = MODEL(arch, is_training=True)

    loss = machine.loss(x_s, y_s, x_t, y_t)
    trainer = TRAINER(loss, arch, args, dirs)
    trainer.train(nIter=arch['training']['max_iter'], machine=machine)


if __name__ == '__main__':
    main()
