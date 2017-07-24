import tensorflow as tf
import pyworld as pw
import soundfile as sf
import numpy as np
import os

from os.path import join


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir_to_wav', None, 'Dir to *.wav')
tf.app.flags.DEFINE_string('dir_to_bin', None, 'Dir to output *.bin')

EPSILON = 1e-10
SPEAKERS = ['SF1', 'SF2', 'SF3', 'SM1', 'SM2',
            'TF1', 'TF2', 'TM1', 'TM2', 'TM3']
FFT_SIZE = 1024
SP_DIM = FFT_SIZE // 2 + 1
FEAT_DIM = SP_DIM + SP_DIM + 1 + 1 + 1  # [sp, ap, f0, en, s]
RECORD_BYTES = FEAT_DIM * 4  # all features saved in `float32`


def wav2pw(x, fs=16000, fft_size=FFT_SIZE):
    ''' Extract WORLD feature from waveform '''
    _f0, t = pw.dio(x, fs)            # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, fs, fft_size=fft_size) # extract aperiodicity
    return {
        'f0': f0,
        'sp': sp,
        'ap': ap,
    }


def extract(filename, fft_size=FFT_SIZE, dtype=np.float32):
    ''' Basic (WORLD) feature extraction ''' 
    x, fs = sf.read(filename, always_2d=True)
    x = x.mean(1)  # to ensure that we can deal with stereo audios
    features = wav2pw(x, fs, fft_size=fft_size)
    ap = features['ap']
    f0 = features['f0'].reshape([-1, 1])
    sp = features['sp']
    en = np.sum(sp + EPSILON, axis=1, keepdims=True)
    sp = np.log10(sp / en)
    return np.concatenate([sp, ap, f0, en], axis=1).astype(dtype)


def extract_and_save_bin_to(dir_to_bin, dir_to_source):
    for d in ['Training Set', 'Testing Set']:
        for s in SPEAKERS:
            path = join(dir_to_source, d, s)
            output_dir = join(dir_to_bin, d, s)
            if not tf.gfile.Exists(output_dir):
                tf.gfile.MakeDirs(output_dir)
            for f in os.listdir(path):
                filename = join(path, f)
                print(filename)
                if not os.path.isdir(filename):
                    features = extract(filename)
                    labels = SPEAKERS.index(s) * np.ones(
                        [features.shape[0], 1],
                        np.float32,
                    )
                    b = os.path.splitext(f)[0]
                    features = np.concatenate([features, labels], 1)
                    with open(join(output_dir, '{}.bin'.format(b)), 'wb') as fp:
                        fp.write(features.tostring())


class Tanhize(object):
    ''' Normalizing `x` to [-1, 1] '''
    def __init__(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.xscale = xmax - xmin
    
    def forward_process(self, x):
        x = (x - self.xmin) / self.xscale
        return tf.clip_by_value(x, 0., 1.) * 2. - 1.

    def backward_process(self, x):
        return (x * .5 + .5) * self.xscale + self.xmin


def read(
    file_pattern,
    batch_size,
    record_bytes=RECORD_BYTES,
    capacity=256,
    min_after_dequeue=128,
    num_threads=8,
    format='NCHW',
    normalizer=None,
    ):
    ''' 
    Read only `sp` and `speaker` 
    Return:
        `feature`: [b, c]
        `speaker`: [b,]
    '''
    with tf.name_scope('InputSpectralFrame'):
        files = tf.gfile.Glob(file_pattern)
        filename_queue = tf.train.string_input_producer(files)


        reader = tf.FixedLengthRecordReader(record_bytes)
        _, value = reader.read(filename_queue)
        value = tf.decode_raw(value, tf.float32)

        value = tf.reshape(value, [FEAT_DIM,])
        feature = value[:SP_DIM]   # NCHW format

        if normalizer is not None:
            feature = normalizer.forward_process(feature)

        if format == 'NCHW':
            feature = tf.reshape(feature, [1, SP_DIM, 1])
        elif format == 'NHWC':
            feature = tf.reshape(feature, [SP_DIM, 1, 1])
        else:
            pass
        speaker = tf.cast(value[-1], tf.int64)
        return tf.train.shuffle_batch(
            [feature, speaker],
            batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            num_threads=num_threads,
            # enqueue_many=True,
        )


def read_whole_features(file_pattern, num_epochs=1):
    '''
    Return
        `feature`: `dict` whose keys are `sp`, `ap`, `f0`, `en`, `speaker`
    '''
    files = tf.gfile.Glob(file_pattern)
    print('{} files found'.format(len(files)))
    filename_queue = tf.train.string_input_producer(files, num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    print("Processing {}".format(key), flush=True)
    value = tf.decode_raw(value, tf.float32)
    value = tf.reshape(value, [-1, FEAT_DIM])
    return {
        'sp': value[:, :SP_DIM],
        'ap': value[:, SP_DIM : 2*SP_DIM],
        'f0': value[:, SP_DIM * 2],
        'en': value[:, SP_DIM * 2 + 1],
        'speaker': tf.cast(value[:, SP_DIM * 2 + 2], tf.int64),
        'filename': key,
    }



def pw2wav(features, feat_dim=513, fs=16000):
    ''' NOTE: Use `order='C'` to ensure Cython compatibility '''
    en = np.reshape(features['en'], [-1, 1])
    sp = np.power(10., features['sp'])
    sp = en * sp
    if isinstance(features, dict):
        return pw.synthesize(
            features['f0'].astype(np.float64).copy(order='C'),
            sp.astype(np.float64).copy(order='C'),
            features['ap'].astype(np.float64).copy(order='C'),
            fs,
        )
    features = features.astype(np.float64)
    sp = features[:, :feat_dim]
    ap = features[:, feat_dim:feat_dim*2]
    f0 = features[:, feat_dim*2]
    en = features[:, feat_dim*2 + 1]
    en = np.reshape(en, [-1, 1])
    sp = np.power(10., sp)
    sp = en * sp
    return pw.synthesize(
        f0.copy(order='C'),
        sp.copy(order='C'),
        ap.copy(order='C'),
        fs
    )

if __name__ == '__main__':
    extract_and_save_bin_to(
        args.dir_to_bin,
        args.dir_to_wav,
        # './dataset/vcc2016/bin',
        # './dataset/vcc2016/wav',
    )
