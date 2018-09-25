import tensorflow as tf
from pprint import pprint

class CPC():
    def __init__(self, X, X_len, Y, Y_label, n=7, code=7, k=2, code_dim=1024, cell_dimension=128):
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
            self.Y_label = Y_label
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
                _, c_t = tf.nn.dynamic_rnn(cell, self.X, initial_state=initial_state)
                #_, c_t = tf.nn.dynamic_rnn(cell, self.X, sequence_length=self.X_len, initial_state=initial_state)
                #c_t = c_t[:, 4, :]
                #c_t = tf.squeeze(c_t, axis=1)
                #raise '18188'
                self.c_t_debug = tf.reduce_mean(c_t)
            self.x_debug = tf.reduce_mean(X)
            print(c_t)

            with tf.variable_scope('coding'):
                predict = []
                losses = []
                y = tf.reshape(self.Y, [self.batch_size, self.k, self.code,  self.code_dim])
                #y = tf.reshape(self.Y, [self.batch_size, self.k * self.code * self.code_dim])
                for i in range(k):
                    W = tf.get_variable('x_t_'+str(i+1), shape=[cell_dimension, self.code * self.code_dim])
                    y_ = tf.reshape(y[:, i, :, :], [self.batch_size, self.code * self.code_dim])

                    cpc = tf.map_fn(lambda x: tf.squeeze(tf.transpose(W) @ tf.expand_dims(x, -1), axis=-1), c_t) * y_
                    #cpc = tf.reshape(cpc, [self.batch_size, self.code, self.code_dim])
                    cpc = tf.reduce_mean(tf.sigmoid(cpc), axis=-1)
                    #cpc = tf.sigmoid(cpc)
                    predict.append(cpc)

                    losses.append(tf.keras.losses.binary_crossentropy(self.Y_label, cpc))

                #predict = tf.concat(predict, axis=1)
                #predict = tf.reshape(predict, [self.batch_size, self.k, self.code * self.code_dim])
                losses = tf.stack(losses, axis=0)
                #losses = tf.reshape(losses, [self.batch_size * self.k])

            # Loss function
            with tf.variable_scope('train'):
                #self.probs = tf.reduce_mean(predict, name='probs', axis=[2,1])
                #self.probs2 = probs2 = tf.reduce_mean(predict, name='probs', axis=2)
                #self.probs = tf.sigmoid(tf.reduce_mean(probs2, axis=[1]))
                self.probs2 = predict#c_t[:, 0:5]#tf.reduce_mean(c_t, axis=-1)
                #self.loss = tf.keras.losses.binary_crossentropy(self.Y_label, self.probs)
                self.loss = tf.reduce_mean(losses)
                #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y_label, logits=self.probs))
