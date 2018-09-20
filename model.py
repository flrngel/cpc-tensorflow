import tensorflow as tf

class CPC():
    def __init__(self, n=7, batch_size=32, cell_dimension=128, d=7*1024):
        """
        Autoregressive part from CPC
        ---
        Placeholder
            X: input (batch_size, fixed_length, vector dimension)

        Argument
            n: padded sequence length
            batch_size: size of batch
            cell_dimension: dimension of cell
            d_row: length of g_encode 
            d_col: dimension of g_encode
        """
        self.n = n
        self.batch_size = batch_size
        self.d = d
        self.X = tf.placeholder(tf.float32, shape=[None, self.n, self.d])
        self.mask = tf.placeholder(tf.bool, shape=[(self.n - 1) * self.batch_size])

        full_sequence = tf.constant(list(range(1, self.n)) * self.batch_size)
        sequence_len = tf.boolean_mask(full_sequence, self.mask)

        with tf.variable_scope('CPC'):
            cell = tf.contrib.rnn.GRUCell(cell_dimension)
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Autoregressive model
            with tf.variable_scope('g_ar'):
                self.ar = tf.nn.dynamic_rnn(cell, self.X, sequence_length=sequence_len, initial_state=initial_state)
                print(self.ar)

            # Loss function
            #with tf.variable_scope('train'):
            #    self.loss = 

CPC()
