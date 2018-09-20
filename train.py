from CPC import CPC

cpc = CPC()

with tf.Session() as sess:
    sess.run(cpc.rnn, 
