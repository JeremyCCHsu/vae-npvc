import numpy as np
import pyworld as pw
import soundfile as sf
import tensorflow as tf
from analyzer import SPEAKERS, pw2wav, read, read_whole_features


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'train_file_pattern',
    './dataset/vcc2016/bin/Training Set/*/*.bin',
    'training dir (to *.bin)')


def main():
    tf.gfile.MkDir('./etc')

    # ==== Save max and min value ====
    x = read_whole_features(args.train_file_pattern)
    x_all = list()
    y_all = list()
    f0_all = list()
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        while True:
            try:
                features = sess.run(x)
                print('Processing {}'.format(features['filename']))
                x_all.append(features['sp'])
                y_all.append(features['speaker'])
                f0_all.append(features['f0'])
            finally:
                pass

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    f0_all = np.concatenate(f0_all, axis=0)


    # ==== F0 stats ====
    for s in SPEAKERS:
        print('Speaker {}'.format(s), flush=True)
        f0 = f0_all[SPEAKERS.index(s) == y_all]
        print('  len: {}'.format(len(f0)))
        f0 = f0[f0 > 2.]
        f0 = np.log(f0)
        mu, std = f0.mean(), f0.std()

        # Save as `float32`
        with open('./etc/{}.npf'.format(s), 'wb') as fp:
            fp.write(np.asarray([mu, std]).tostring())


    # ==== Min/Max value ====
    # mu = x_all.mean(0)
    # std = x_all.std(0)
    q005 = np.percentile(x_all, 0.5, axis=0)
    q995 = np.percentile(x_all, 99.5, axis=0)

    # Save as `float32`
    with open('./etc/xmin.npf', 'wb') as fp:
        fp.write(q005.tostring())

    with open('./etc/xmax.npf', 'wb') as fp:
        fp.write(q995.tostring())



def test():
    # ==== Test: batch mixer (conclusion: capacity should be larger to make sure good mixing) ====
    x, y = read('./dataset/vcc2016/bin/*/*/1*001.bin', 32, min_after_dequeue=1024, capacity=2048)
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        for _ in range(200):
            x_, y_ = sess.run([x, y])
            print(y_)


    # ===== Read binary ====
    features = read_whole_features('./dataset/vcc2016/bin/Training Set/SF1/*001.bin')

    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        features = sess.run(features)

    y = pw2wav(features)
    sf.write('test1.wav', y, 16000)  # TODO fs should be specified externally.


    # ==== Direct read =====
    f = './dataset/vcc2016/bin/Training Set/SF1/100001.bin'
    features = np.fromfile(f, np.float32)
    features = np.reshape(features, [-1, 513*2 + 1 + 1 + 1]) # f0, en, spk

    y = pw2wav(features)
    sf.write('test2.wav', y, 16000)


if __name__ == '__main__':
    main()
