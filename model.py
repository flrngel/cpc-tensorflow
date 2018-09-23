import tensorflow as tf
from pprint import pprint

class CPC():
    def __init__(self, X, X_len, Y, n=7, code=7, k=2, code_dim=1024, cell_dimension=128):
        """
        Autoregressive part from CPC
        ---
        tf.placeholder
            X: input (batch_size, fixed_length, vector dimension)

        Argument
            n: padded sequence length
        """

        with tf.variable_scope('CPC'):
            self.X = X
            self.X_len = X_len
            self.Y = Y
            self.batch_size = X.shape[0]
            #self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

            pprint({'X': X, 'X_len': X_len, 'Y': Y, 'batch_size': self.batch_size})

            self.n = n
            self.k = k
            self.code = code
            self.code_dim = code_dim
            self.cell_dimension = cell_dimension

            #self.X = tf.placeholder(tf.float32, shape=[None, self.n * self.code, self.code_dim], name='X')
            #self.Y = tf.placeholder(tf.float32, shape=[None, self.k * self.code, self.code_dim], name='Y')
            #self.X_len = tf.placeholder(tf.float32, shape=[None], name='X_len')

            cell = tf.contrib.rnn.GRUCell(cell_dimension, name='cell')
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Autoregressive model
            with tf.variable_scope('g_ar'):
                _, c_t = tf.nn.dynamic_rnn(cell, self.X, sequence_length=self.X_len, initial_state=initial_state)

            with tf.variable_scope('coding'):
                predict = []
                y = tf.reshape(self.Y, [self.batch_size, self.k * self.code * self.code_dim])
                for i in range(k):
                    cpc = tf.layers.dense(c_t, k * code * code_dim, name='x_t_'+str(i+1))
                    out = tf.sigmoid(y * cpc)
                    predict.append(out)
                predict = tf.transpose(predict, perm=[1, 0, 2])

            # Loss function
            with tf.variable_scope('train'):
                self.loss = tf.reduce_mean(tf.subtract(1.0, predict), name='loss', axis=[2,1,0])
