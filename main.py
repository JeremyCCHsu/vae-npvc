import os
import json

import tensorflow as tf
import numpy as np

# from model.vae import VAWGAN
from analyzer import read, Tanhize
from util.wrapper import save, validate_log_dirs #, load, configure_gpu_settings, restore_global_step
# from trainer.vae import VAWGANTrainer
from importlib import import_module

args = tf.app.flags.FLAGS
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

tf.app.flags.DEFINE_string('model_module', 'model.vae', 'Model module')
tf.app.flags.DEFINE_string('model', None, 'Model: ConvVAE, VAWGAN')

tf.app.flags.DEFINE_string('trainer_module', 'trainer.vae', 'Trainer module')
tf.app.flags.DEFINE_string('trainer', None, 'Trainer: VAETrainer, VAWGANTrainer')

if args.model is None or args.trainer is None:
    raise ValueError(
        '\n  Both `model` and `trainer` should be assigned.' +\
        '\n  Use `python main.py --help` to see applicable options.'
    )

module = import_module(args.model_module, package=None)
MODEL = getattr(module, args.model)

module = import_module(args.trainer_module, package=None)
TRAINER = getattr(module, args.trainer)

def main():
    ''' NOTE: The input is rescaled to [-1, 1] '''

    dirs = validate_log_dirs(args)
    tf.gfile.MakeDirs(dirs['logdir'])

    with open(args.architecture) as f:
        arch = json.load(f)

    with open(os.path.join(dirs['logdir'], args.architecture), 'w') as f:
        json.dump(arch, f, indent=4)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/xmax.npf'),
        xmin=np.fromfile('./etc/xmin.npf'),
    )

    image, label = read(
        file_pattern=arch['training']['datadir'],
        batch_size=arch['training']['batch_size'],
        capacity=2048,
        min_after_dequeue=1024,
        normalizer=normalizer,
    )

    machine = MODEL(arch)

    loss = machine.loss(image, label)
    trainer = TRAINER(loss, arch, args, dirs)
    trainer.train(nIter=arch['training']['max_iter'], machine=machine)


if __name__ == '__main__':
    main()
