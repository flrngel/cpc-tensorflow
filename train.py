import tensorflow as tf
import numpy as np
from tensorflow import keras
from model import CPC
from nets.resnet_v2 import resnet_v2_101 as resnet

learn_rate = 2e-4
batch_size = 32

def image_preprocess(x):
    x = tf.expand_dims(x, axis=-1)
    x = tf.concat([x, x, x], axis=-1)
    x = tf.image.resize_images(x, (224, 224))
    return x

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

# load data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
batches = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)
iterator = batches.make_one_shot_iterator()
items = iterator.get_next()
data = image_preprocess(items)

# build graph
## resnet
_, features = resnet(data)
features = features['resnet_v2_101/block3']
Y = features[:, -2:, : , :]
X = tf.reshape(features, shape=[batch_size, 7 * 7, 1024])

## cpc
X_len = [5] * batch_size
X_len = tf.constant(X_len, dtype=tf.int32)

cpc = CPC(X, X_len, Y)
train_op = tf.train.AdamOptimizer(learn_rate).minimize(cpc.loss)

# tensorflow
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    step = 0

    while True:
        try:
            _, loss = sess.run([train_op, cpc.loss])
            step += 1
            if step % 1000 == 0:
                print(loss)
        except tf.errors.OutOfRangeError:
            break
