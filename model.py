import tensorflow as tf

class CPC():
    def __init__(self, n=7, col=7, code_size=1024, cell_dimension=128):
        """
        Autoregressive part from CPC
        ---
        tf.placeholder
            X: input (batch_size, fixed_length, vector dimension)

        Argument
            n: padded sequence length
        """
        self.n = n
        self.col = col
        self.code_size = code_size
        self.cell_dimension = cell_dimension

        self.X = tf.placeholder(tf.float32, shape=[None, self.n * self.col, self.code_size], name='X')
        assert self.X.shape[1:] == (self.n * self.col, self.code_size)
        self.X_len = tf.placeholder(tf.float32, shape=[None], name='X_len')
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

        with tf.variable_scope('CPC'):
            cell = tf.contrib.rnn.GRUCell(cell_dimension)
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Autoregressive model
            with tf.variable_scope('g_ar'):
                _, c_t = tf.nn.dynamic_rnn(cell, self.X, sequence_length=self.X_len, initial_state=initial_state)
                print(c_t)

            # Loss function
            #with tf.variable_scope('train'):
            #    self.loss = 

CPC()
