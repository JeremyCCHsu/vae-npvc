import json
import os

import numpy as np
import soundfile as sf
import tensorflow as tf
from analyzer import SPEAKERS, Tanhize, pw2wav, read_whole_features
from model.vae import ConvVAE
from util.wrapper import load
# from util.image import gray2jet


def convert_f0(f0, src, trg):
    mu_s, std_s = np.fromfile(os.path.join('./etc', '{}.npf'.format(src)), np.float32)
    mu_t, std_t = np.fromfile(os.path.join('./etc', '{}.npf'.format(trg)), np.float32)
    lf0 = tf.where(f0 > 1., tf.log(f0), f0)
    lf0 = tf.where(lf0 > 1., (lf0 - mu_s)/std_s * std_t + mu_t, lf0)
    lf0 = tf.where(lf0 > 1., tf.exp(lf0), lf0)
    return lf0


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint', None, 'root of log dir')
tf.app.flags.DEFINE_string('src', 'SF1', 'source speaker [SF1 - SM2]')
tf.app.flags.DEFINE_string('trg', 'TM3', 'target speaker [SF1 - TM3]')

FS = 16000

def main():
    logdir, ckpt = os.path.split(args.checkpoint)
    arch = tf.gfile.Glob(os.path.join(logdir, 'architecture*.json'))[0]  # should only be 1 file
    with open(arch) as fp:
        arch = json.load(fp)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/xmax.npf'),
        xmin=np.fromfile('./etc/xmin.npf'),
    )

    features = read_whole_features('./dataset/vcc2016/bin/Testing Set/{}/200001.bin'.format(args.src))

    y_s = features['speaker']
    x = normalizer.forward_process(features['sp'])
    x = tf.expand_dims(x, 1)   # [b, h] => [b, c=1, h]
    x = tf.expand_dims(x, -1)  # => [b, c=1, h, w=1]
    # import pdb; pdb.set_trace()
    y_t = SPEAKERS.index(args.trg) * tf.ones(shape=[tf.shape(x)[0],], dtype=tf.int64)

    machine = ConvVAE(arch, is_training=True)
    z = machine.encode(x)
    x_t = machine.decode(z, y_t)  # NOTE: the API yields NHWC format
    x_t = tf.squeeze(x_t)
    x_t = normalizer.backward_process(x_t)

    # for sanity check (validation)
    x_s = machine.decode(z, y_s)
    x_s = tf.squeeze(x_s)
    x_s = normalizer.backward_process(x_s)

    f0_s = features['f0']
    f0_t = convert_f0(f0_s, args.src, args.trg)

    # TODO: add file loop, src loop, trg loop

    saver = tf.train.Saver()
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        load(saver, sess, logdir, ckpt=ckpt)

        feat, f0, sp = sess.run([features, f0_t, x_t])
        # import pdb; pdb.set_trace()
        feat['sp'] = sp
        feat['f0'] = f0
        y = pw2wav(feat)
        sf.write('test-{}-{}.wav'.format(args.src, args.trg), y, FS)

        # converted, reconst = sess.run([x_t, x_s])
        # with tf.gfile.GFile('test-conv.png', 'wb') as fp:
        #     fp.write(converted)

        # with tf.gfile.GFile('test-reconst.png', 'wb') as fp:
        #     fp.write(reconst)

    # TODO: validator should be periodically executed during training. 

if __name__ == '__main__':
    main()
