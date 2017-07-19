import tensorflow as tf


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def make_png_thumbnail(x, n):
    '''
    Input:
        `x`: Tensor, value range=[-1, 1), shape=[n*n, h, w, c]
        `n`: sqrt of the number of images
    
    Return:
        `tf.string` (bytes) of the PNG. 
        (write these binary directly into a file)
    '''
    with tf.name_scope('MakeThumbnail'):
        _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, [n, n, h, w, c])
        x = tf.transpose(x, [0, 2, 1, 3, 4])
        x = tf.reshape(x, [n * h, n * w, c])
        x = x / 2. + .5
        x = tf.image.convert_image_dtype(x, tf.uint8, saturate=True)
        x = tf.image.encode_png(x)
    return x


def line(x, xa, xb, ya, yb):
    ''' a line determined by two points '''
    return ya + (x - xa) * (yb - ya) / (xb - xa)


def clip_to_boundary(line1, line2, minval, maxval):
    x = tf.minimum(line1, line2)
    x = tf.minimum(x, maxval)
    x = tf.maximum(x, minval)
    return x


def gray2jet(x):
    ''' NHWC (channel last) format '''
    with tf.name_scope('Gray2Jet'):
        r = clip_to_boundary(
            line(x, .3515, .66, 0., 1.),
            line(x, .8867, 1., 1., .5),
            minval=0.,
            maxval=1.,
        )
        g = clip_to_boundary(
            line(x, .125, .375, 0., 1.),
            line(x, .64, .91, 1., 0.),
            minval=0.,
            maxval=1.,
        )
        b = clip_to_boundary(
            line(x, .0, .1132, 0.5, 1.0),
            line(x, .34, .648, 1., 0.),
            minval=0.,
            maxval=1.,
        )
        return tf.concat([r, g, b], axis=-1)


def make_png_jet_thumbnail(x, n):
    '''
    Input:
        `x`: Tensor, value range=[-1, 1), shape=[n*n, h, w, c]
        `n`: sqrt of the number of images
    
    Return:
        `tf.string` (bytes) of the PNG. 
        (write these binary directly into a file)
    '''
    with tf.name_scope('MakeThumbnail'):
        _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, [n, n, h, w, c])
        x = tf.transpose(x, [0, 2, 1, 3, 4])
        x = tf.reshape(x, [n * h, n * w, c])
        x = x / 2. + .5
        x = gray2jet(x)
        x = tf.image.convert_image_dtype(x, tf.uint8, saturate=True)
        x = tf.image.encode_png(x)
    return x