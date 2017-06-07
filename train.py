import json
import os
import pdb
import sys

import numpy as np
import tensorflow as tf

from models import MLPcVAE
from trainers import VAETrainer
from vcc2016io import VCC2016TFRManager
from util.wrapper import validate_log_dirs

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'datadir', '/home/jrm/proj/vc2016b/TR_log_SP_Z_LT8000', 'data dir')
tf.app.flags.DEFINE_string(
    'architecture', 'architecture.json', 'network architecture')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'resotre form dir')
tf.app.flags.DEFINE_integer('summary_freq', 1000, 'summary frequency')

def main():
    dirs = validate_log_dirs(args)
    tf.gfile.MakeDirs(dirs['logdir'])

    with open(args.architecture) as f:
        arch = json.load(f)

    with open(os.path.join(dirs['logdir'], args.architecture), 'w') as f:
        json.dump(arch, f, indent=4)

    # Data/Batch
    with tf.name_scope('create_input'):
        reader = VCC2016TFRManager()
        x, y = reader.read(
            file_pattern=arch['training']['file_pattern'],
            batch_size=arch['training']['batch_size'],
        )

    net = MLPcVAE(arch=arch, is_training=True)
    loss = net.loss(x, y)
    trainer = VAETrainer(loss, arch, args, dirs)
    trainer.train(nIter=arch['training']['max_iter'])

if __name__ == "__main__":
    main()
