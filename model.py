import tensorflow as tf

class CPC():
    def __init__(self, X, X_len, Y, Y_label, n=7, code=7, k=2, code_dim=1024, cell_dimension=128):
        """
        Autoregressive part from CPC
        """

        with tf.variable_scope('CPC'):
            self.X = X
            self.X_len = X_len
            self.Y = Y
            self.Y_label = Y_label
            self.batch_size = X.shape[0]


            self.n = n
            self.k = k
            self.code = code
            self.code_dim = code_dim
            self.cell_dimension = cell_dimension

            cell = tf.contrib.rnn.GRUCell(cell_dimension, name='cell')
            initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # Autoregressive model
            with tf.variable_scope('g_ar'):
                _, c_t = tf.nn.dynamic_rnn(cell, self.X, sequence_length=self.X_len, initial_state=initial_state)
                self.c_t = c_t

            with tf.variable_scope('coding'):
                losses = []
                y = self.Y
                for i in range(k):
                    W = tf.get_variable('x_t_'+str(i+1), shape=[cell_dimension, self.code_dim])
                    y_ = tf.reshape(y[:, i, :], [self.batch_size, self.code_dim])
                    self.cpc = tf.map_fn(lambda x: tf.squeeze(tf.transpose(W) @ tf.expand_dims(x, -1), axis=-1), c_t) * y_
                    nce = tf.sigmoid(tf.reduce_mean(self.cpc, -1))
                    losses.append(tf.keras.losses.binary_crossentropy(self.Y_label, nce))

                losses = tf.stack(losses, axis=0)

            # Loss function
            with tf.variable_scope('train'):
                self.loss = tf.reduce_mean(losses)
