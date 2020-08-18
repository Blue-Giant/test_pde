import tensorflow as tf

import numpy as np

a_array = np.array([[1, 2, 3], [4, 5, 6]])
b_list = [[1, 2, 3], [3, 4, 5]]
c_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])

with tf.Session() as sess:
    shapeA = tf.shape(a_array)
    dim = tf.rank(shapeA)
    print(sess.run(shapeA))
    print(sess.run(tf.shape(b_list)))
    shapeC = c_tensor.get_shape()
    dim = len(shapeC)
    rankC = tf.rank(c_tensor)
    print(rankC)
    shapeCC = rankC.get_shape()
    print(shapeCC)
    if rankC==1:
        print(rankC)
    if rankC==2:
        print(rankC)
    if rankC==3:
        print('fghjklkkjhgfghjklkjhj')
        print(sess.run(rankC))