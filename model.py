import tensorflow as tf

class CPC():
    def __init__(self, cell_dimension=128, n=7, d=7):
        """
        Autoregressive part from CPC
        ---
        Placeholder
            X: input (batch_size, fixed_length, vector dimension)

        Argument
            cell_dimension: dimension of cell
            n: length of g_encode 
            d: dimension of g_encode
        """
        self.n = n
        self.d = d
        self.X = tf.placeholder(tf.float32, shape=[None, self.n, self.d])
        with tf.variable_scope('CPC'):
            assert len(self.X.shape()) == 3, "input should be (batch_size, fixed_length, vector dimension)"

            self.n = self.X.shape()[1]
            self.batch_size = self.X.shape()[0]

            
            cell = tf.contrib.rnn.GRUCell(cell_dimension)
            initial_state = cell.zero_stae(self.batch_size, dypte=tf.float32)

            # Autoregressive model
            with tf.variable_scope('g_ar'):
                self.ar = tf.nn.dynamic_rnn(cell, self.X, sequence_length=[range(n)] * self.batch_size)
