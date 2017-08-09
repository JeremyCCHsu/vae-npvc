import json
import os

import tensorflow as tf
import numpy as np
import soundfile as sf

from util.wrapper import load
from analyzer import read_whole_features, SPEAKERS, pw2wav
from analyzer import Tanhize
from datetime import datetime
from importlib import import_module

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint', None, 'root of log dir')
tf.app.flags.DEFINE_string('src', 'SF1', 'source speaker [SF1 - SM2]')
tf.app.flags.DEFINE_string('trg', 'TM3', 'target speaker [SF1 - TM3]')
tf.app.flags.DEFINE_string('output_dir', './logdir', 'root of output dir')
tf.app.flags.DEFINE_string('module', 'model.vae', 'Module')
tf.app.flags.DEFINE_string('model', None, 'Model')
tf.app.flags.DEFINE_string('file_pattern', './dataset/vcc2016/bin/Testing Set/{}/*.bin', 'file pattern')

if args.model is None:
    raise ValueError(
        '\n  You MUST specify `model`.' +\
        '\n    Use `python convert.py --help` to see applicable options.'
    )

module = import_module(args.module, package=None)
MODEL = getattr(module, args.model)

FS = 16000


def make_output_wav_name(output_dir, filename):
    basename = str(filename, 'utf8')
    basename = os.path.split(basename)[-1]
    basename = os.path.splitext(basename)[0]
    print('Processing {}'.format(basename))        
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


def main():
    logdir, ckpt = os.path.split(args.checkpoint)
    arch = tf.gfile.Glob(os.path.join(logdir, 'architecture*.json'))[0]  # should only be 1 file
    with open(arch) as fp:
        arch = json.load(fp)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/xmax.npf'),
        xmin=np.fromfile('./etc/xmin.npf'),
    )

    features = read_whole_features(args.file_pattern.format(args.src))

    x = normalizer.forward_process(features['sp'])
    x = nh_to_nchw(x)
    y_s = features['speaker']
    y_t_id = tf.placeholder(dtype=tf.int64, shape=[1,])
    y_t = y_t_id * tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)

    machine = MODEL(arch)
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
        while True:
            try:
                feat, f0, sp = sess.run(
                    [features, f0_t, x_t],
                    feed_dict={y_t_id: np.asarray([SPEAKERS.index(args.trg)])}
                )
                feat.update({'sp': sp, 'f0': f0})
                y = pw2wav(feat)
                oFilename = make_output_wav_name(output_dir, feat['filename'])
                sf.write(oFilename, y, FS)
            except:
                break

if __name__ == '__main__':
    main()
