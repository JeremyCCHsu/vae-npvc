import tensorflow as tf

FEAT_DIM = 513
LABEL_BYTES = 4
FRAME_BYTES = FEAT_DIM * 4
RECORD_BYTES = LABEL_BYTES + FRAME_BYTES


class VCC2016TFRManager(object):
    ''' Read tensor batch from VCC2016 TF records '''
    def __init__(self, ftype='log-spectrum'):
        self.n_files = None
        self.ftype = ftype
    
    
    def read(
        self,
        file_pattern,
        batch_size,
        num_preprocess_threads=10,
        capacity=512,
        min_after_dequeue=256,
        ):
        '''
        Return:
            `frame`: shape = [b, c]
            `label`: shape = [b,]  
        '''

        with tf.variable_scope('input'):
            files = tf.gfile.Glob(file_pattern)
            self.n_files = len(files)
            filename_queue = tf.train.string_input_producer(files)

            reader = tf.FixedLengthRecordReader(RECORD_BYTES)
            _, value = reader.read(filename_queue)
            floats = tf.decode_raw(value, tf.float32, little_endian=True)
            frame = tf.reshape(floats[1: 1 + FEAT_DIM], [1, FEAT_DIM])

            if self.ftype == 'spectrum':
                frame = tf.pow(10.0, frame)

            label = tf.reshape(floats[0], [1,])
            label = tf.cast(label, tf.int64)
            return tf.train.shuffle_batch(            
                [frame, label],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=capacity,
                enqueue_many=True,
                min_after_dequeue=min_after_dequeue)


    def read_whole(self, file_pattern, num_epochs=None):
        '''
        Return:
            `filename`
            `frame`: shape = [None, FEAT_DIM]
            `label`: shape = [None,]
        '''
        files = tf.gfile.Glob(file_pattern)
        self.n_files = len(files)
        print('found {} files'.format(self.n_files))
        filename_queue = tf.train.string_input_producer(
            files,
            num_epochs=num_epochs)

        reader = tf.WholeFileReader()
        filename, value = reader.read(filename_queue)
        features = tf.decode_raw(value, tf.float32)
        features = tf.reshape(features, [-1, FEAT_DIM + 1])  

        label = tf.cast(features[:, 0], tf.int64)
        frame = features[:, 1:]

        if self.ftype == 'spectrum':
            frame = tf.pow(10.0, frame)
        
        return {
            'filename': filename,
            'frame': frame,
            'label': label,
        }
