import json
import os
import sys

import tensorflow as tf
import numpy as np
import soundfile as sf

from util.wrapper import load
from analyzer import read_whole_features, pw2wav
from analyzer import Tanhize
from datetime import datetime
from importlib import import_module

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('corpus_name', 'vcc2016', 'Corpus name')
tf.app.flags.DEFINE_string('checkpoint', None, 'root of log dir')
tf.app.flags.DEFINE_string('src', 'SF1', 'source speaker [SF1 - SM2]')
tf.app.flags.DEFINE_string('trg', 'TM3', 'target speaker [SF1 - TM3]')
tf.app.flags.DEFINE_string('output_dir', './logdir', 'root of output dir')
tf.app.flags.DEFINE_string('module', 'model.vawgan', 'Module')
tf.app.flags.DEFINE_string('model', 'VAWGAN', 'Model')
tf.app.flags.DEFINE_string('file_pattern', './dataset/vcc2016/bin/Testing Set/{}/*.bin', 'file pattern')
tf.app.flags.DEFINE_string(
    'speaker_list', './etc/speakers.tsv', 'Speaker list (one speaker per line)'
)

def make_output_wav_name(output_dir, filename):
    basename = str(filename, 'utf8')
    basename = os.path.split(basename)[-1]
    basename = os.path.splitext(basename)[0]
    return os.path.join(
        output_dir, 
        '{}-{}-{}.wav'.format(args.src, args.trg, basename)
    )

def get_default_output(logdir_root):
    STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
    logdir = os.path.join(logdir_root, 'output', STARTED_DATESTRING)
    print('Using default logdir: {}'.format(logdir))        
    return logdir

def convert_f0(f0, src, trg):
    mu_s, std_s = np.fromfile(os.path.join('./etc', '{}.npf'.format(src)), np.float32)
    mu_t, std_t = np.fromfile(os.path.join('./etc', '{}.npf'.format(trg)), np.float32)
    lf0 = tf.where(f0 > 1., tf.log(f0), f0)
    lf0 = tf.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    lf0 = tf.where(lf0 > 1., tf.exp(lf0), lf0)
    return lf0


def nh_to_nchw(x):
    with tf.name_scope('NH_to_NCHW'):
        x = tf.expand_dims(x, 1)      # [b, h] => [b, c=1, h]
        return tf.expand_dims(x, -1)  # => [b, c=1, h, w=1]

def nh_to_nhwc(x):
    with tf.name_scope('NH_to_NHWC'):
        return tf.expand_dims(tf.expand_dims(x, -1), -1)


def main(unused_args=None):
    # args(sys.argv)

    if args.model is None:
        raise ValueError(
            '\n  You MUST specify `model`.' +\
            '\n    Use `python convert.py --help` to see applicable options.'
        )

    module = import_module(args.module, package=None)
    MODEL = getattr(module, args.model)

    FS = 16000

    with open(args.speaker_list) as fp:
        SPEAKERS = [l.strip() for l in fp.readlines()]


    logdir, ckpt = os.path.split(args.checkpoint)
    if 'VAE' in logdir:
        _path_to_arch, _ = os.path.split(logdir)
    else:
        _path_to_arch = logdir
    arch = tf.gfile.Glob(os.path.join(_path_to_arch, 'architecture*.json'))
    if len(arch) != 1:
        print('WARNING: found more than 1 architecture files!')
    arch = arch[0]
    with open(arch) as fp:
        arch = json.load(fp)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/{}_xmax.npf'.format(args.corpus_name)),
        xmin=np.fromfile('./etc/{}_xmin.npf'.format(args.corpus_name)),
    )

    features = read_whole_features(args.file_pattern.format(args.src))

    x = normalizer.forward_process(features['sp'])
    x = nh_to_nhwc(x)
    y_s = features['speaker']
    y_t_id = tf.placeholder(dtype=tf.int64, shape=[1,])
    y_t = y_t_id * tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)

    machine = MODEL(arch, is_training=False)
    z = machine.encode(x)
    x_t = machine.decode(z, y_t)  # NOTE: the API yields NHWC format
    x_t = tf.squeeze(x_t)
    x_t = normalizer.backward_process(x_t)

    # For sanity check (validation)
    x_s = machine.decode(z, y_s)
    x_s = tf.squeeze(x_s)
    x_s = normalizer.backward_process(x_s)

    f0_s = features['f0']
    f0_t = convert_f0(f0_s, args.src, args.trg)

    output_dir = get_default_output(args.output_dir)

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(logdir=output_dir)
    with sv.managed_session() as sess:
        load(saver, sess, logdir, ckpt=ckpt)
        print()
        while True:
            try:
                feat, f0, sp = sess.run(
                    [features, f0_t, x_t],
                    feed_dict={y_t_id: np.asarray([SPEAKERS.index(args.trg)])}
                )
                feat.update({'sp': sp, 'f0': f0})
                y = pw2wav(feat)
                oFilename = make_output_wav_name(output_dir, feat['filename'])
                print('\rProcessing {}'.format(oFilename), end='')
                sf.write(oFilename, y, FS)
            except KeyboardInterrupt:
                break
            finally:
                pass
        print()

if __name__ == '__main__':
    tf.app.run()
