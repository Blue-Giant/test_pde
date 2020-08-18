import tensorflow as tf
import numpy as np

XY_it = tf.placeholder(tf.float32, name='X_it', shape=[None, 3])
U_hat = tf.reshape(tf.gather(XY_it, [0], axis=-1), shape=[-1, 1])
Psi_hat2X = tf.reshape(tf.gather(XY_it, [1], axis=-1), shape=[-1, 1])
Psi_hat2Y = tf.reshape(tf.gather(XY_it, [2], axis=-1), shape=[-1, 1])
Psi_hat = tf.concat([Psi_hat2X, Psi_hat2Y], axis=-1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xy = np.array([[1,2,3]])
    U,PX,PY,PXY = sess.run([U_hat, Psi_hat2X, Psi_hat2Y, Psi_hat], feed_dict={XY_it:xy})
    print('u:',U)
    print('px:', PX)
    print('py:', PY)
    print('pxy:', PXY)