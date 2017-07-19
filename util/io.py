import numpy as np
import tensorflow as tf

def int64_feature(value):
    if isinstance(value, list):
        value = [tf.train.Feature(
            int64_list=tf.train.Int64List(value=[v])) for v in value]
        return tf.train.FeatureList(feature=value)
    else:
        value = tf.train.Int64List(value=[value])
        return tf.train.Feature(int64_list=value)


def bytes_feature(value):
    if isinstance(value, list):
        value = [tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[v])) for v in value]
        return tf.train.FeatureList(feature=value)
    else:
        value = tf.train.BytesList(value=[value])
        return tf.train.Feature(bytes_list=value)


def float_feature(value):
    if isinstance(value, list):
        value = [tf.train.Feature(
            float_list=tf.train.FloatList(value=[v])) for v in value]
        return tf.train.FeatureList(feature=value)
    else:
        value = tf.train.FloatList(value=[value])
        return tf.train.Feature(float_list=value)


def read_float64_as_float32(filename):
    x = np.fromfile(filename, np.float64)
    return x.astype(np.float32)
