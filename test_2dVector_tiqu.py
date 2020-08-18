import tensorflow as tf
import numpy as np
input_dim = 2
X_left_bd = tf.placeholder(tf.float32, name='X_left_bd', shape=[None, input_dim])
temp0 = X_left_bd[:, 0]
temp1 = X_left_bd[:, 1]
y = 2*X_left_bd

with tf.Session() as sess:
    xl_bd_batch = [[0, 1], [2, 3], [4, 5], [6, 7]]
    xl_bd_batch = np.asarray(xl_bd_batch)
    print(xl_bd_batch)
    Y, T0, T1 = sess.run([y, temp0, temp1], feed_dict={X_left_bd: xl_bd_batch})
    print('T0:', T0)
    print('T1:', T1)
    print('Y:',Y)