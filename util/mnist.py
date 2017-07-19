import tensorflow as tf
from tensorflow.contrib import keras


def mnist_batcher_in_tanh_vector(
    batch_size,
    capacity=256,
    min_after_dequeue=128,
    ):
    (x, y), (_, _) = keras.datasets.mnist.load_data()
    x = tf.constant(x)
    x = tf.cast(x, tf.float32)
    x = keras.layers.Flatten()(x) / 127.5 - 1.
    y = tf.cast(y, tf.int64)

    return tf.train.shuffle_batch(
        [x, y],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True
    )


def mnist16_batcher_in_tanh(
    batch_size,
    capacity=256,
    min_after_dequeue=128,
    ):
    '''
    NCHW (channel first)
    '''
    with tf.name_scope('MNIST_nchw'):
        (x, y), (_, _) = keras.datasets.mnist.load_data()
        x = x / 127.5 - 1.
        x = tf.constant(x, dtype=tf.float32)
        x = tf.expand_dims(x, -1)  # n, h, w, c
        x = tf.image.resize_nearest_neighbor(x, [16, 16])
        x = tf.transpose(x, [0, 3, 1, 2])

        x = tf.Variable(x, trainable=False, name='image', dtype=tf.float32)
        y = tf.Variable(y, trainable=False, name='label', dtype=tf.int64)

        # TODO: using `slice_input_producer` slows down training!!! WHY?
        # 9.7 steps per sec -> 5.7 steps per sec
        # (but it consumes more GPU memory and GPU-Util, why?!)

        x, y = tf.train.slice_input_producer([x, y]) #, shuffle=False)

        return tf.train.batch(
        # return tf.train.shuffle_batch(
            [x, y],
            batch_size=batch_size,
            num_threads=8,
            # capacity=capacity,
            # min_after_dequeue=min_after_dequeue,
            # enqueue_many=True
        )

def mnist16_batcher_in_z_norm(
    batch_size,
    capacity=256,
    min_after_dequeue=128,
    ):
    '''
    NCHW (channel first)
    '''
    (x, y), (_, _) = keras.datasets.mnist.load_data()
    x_mu = x.mean(0)
    x_std = x.std(0)
    x = (x - x_mu) / (x_std + 1e-6)
    y = tf.constant(y, dtype=tf.int64)
    # x = tf.constant(x, dtype=tf.float32) / 127.5 - 1.
    x = tf.constant(x, dtype=tf.float32)
    x = tf.clip_by_value(x, -5, 5)
    x = tf.expand_dims(x, -1)  # n, h, w, c
    x = tf.image.resize_bilinear(x, [16, 16])
    x = tf.transpose(x, [0, 3, 1, 2])

    return tf.train.shuffle_batch(
        [x, y],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=True
    )
