import json
import os
import sys
from datetime import datetime

import tensorflow as tf


def get_checkpoint(logdir):
    ''' Get the first checkpoint '''
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        return ckpt.model_checkpoint_path
    else:
        print('No checkpoint found')
        return None

def save(saver, sess, logdir, step):
    ''' Save a model to logdir/model.ckpt-[step] '''
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir, ckpt=None):
    '''
    Try to load model form a dir (search for the newest checkpoint)
    '''
    print('Trying to restore checkpoints from {} ...'.format(logdir),
        end="")
    if ckpt:
        ckpt = os.path.join(logdir, ckpt)
        global_step = int(
            ckpt
            .split('/')[-1]
            .split('-')[-1])
        print('  Global step: {}'.format(global_step))
        print('  Restoring...', end="")
        saver.restore(sess, ckpt)
        return global_step
    else:
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
            global_step = int(
                ckpt.model_checkpoint_path
                .split('/')[-1]
                .split('-')[-1])
            print('  Global step: {}'.format(global_step))
            print('  Restoring...', end="")
            saver.restore(sess, ckpt.model_checkpoint_path)
            return global_step
        else:
            print('No checkpoint found')
            return None

# 

def configure_gpu_settings(gpu_cfg=None):
    session_conf = None
    if gpu_cfg:
        with open(gpu_cfg) as f:
            cfg = json.load(f)
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=cfg['per_process_gpu_memory_fraction'])
        session_conf = tf.ConfigProto(
            allow_soft_placement=cfg['allow_soft_placement'],
            log_device_placement=cfg['log_device_placement'],
            inter_op_parallelism_threads=cfg['inter_op_parallelism_threads'],
            intra_op_parallelism_threads=cfg['intra_op_parallelism_threads'],
            gpu_options=gpu_options)
        # Timeline
        # jit_level = 0
        # session_conf.graph_options.optimizer_options.global_jit_level = jit_level
    #     sess = tf.Session(
    #         config=session_conf)
    # else:
    #     sess = tf.Session()
    return session_conf

def restore_global_step(saver, sess, from_dir, ckpt):
    try:
        step = load(saver, sess, from_dir, ckpt)
        if step is None:
            step = 0
    except:
        print("Something's wrong while restoing checkpoints!")
        raise
    return step


def validate_log_dirs(args):
    ''' Create a default log dir (if necessary) '''
    def get_default_logdir(logdir_root):
        STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
        logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
        print('Using default logdir: {}'.format(logdir))        
        return logdir

    if args.logdir and args.restore_from:
        raise ValueError(
            'You can only specify one of the following: ' +
            '--logdir and --restore_from')

    if args.logdir and args.log_root:
        raise ValueError(
            'You can only specify either --logdir or --logdir_root')

    if args.logdir_root is None:
        logdir_root = 'logdir'

    if args.logdir is None:
        logdir = get_default_logdir(logdir_root)

    # Note: `logdir` and `restore_from` are exclusive
    if args.restore_from is None:
        restore_from = logdir
    else:
        restore_from = args.restore_from

    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from,
    }
