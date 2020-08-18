import numpy as np
import tensorflow as tf

# "it" for interior
def rand_it(size, input_dim):
    x_it = np.random.rand(size, input_dim)
    x_it = x_it.astype(np.float32)
    return x_it


def neural_fun(input_x, in_size, n_hidden):
    W = tf.Variable(tf.random_uniform([in_size, n_hidden]), dtype='float32', name='W-transInput')
    B = tf.Variable(tf.ones([1, n_hidden]), dtype='float32', name='B-transInput')
    out = tf.add(tf.matmul(input_x, W), B)
    return out
#  输出的结果为 1 X 自变量的个数。是一个列表，如果关于自变量求导，可以看做多元函数微分，导数值会求和
#  输入 [x1,x2], 权重为 2行5列的全1矩阵，那么得到的结果为
#  out = [x1+x2,x1+x2,x1+x2,x1+x2,x1+x2]
#  然后 out 关于 x1 和 x2 求导，就是多个函数关于多个变量求导。
#  out/x1 = 1+1+1+1+1 = 5
#  out/x2 = 1+1+1+1+1 = 5

dim =2
batchsize_it = 1
layer_wide = 3

X_it = tf.placeholder(tf.float32, name='X_it', shape=[None, dim])
u = neural_fun(X_it, dim, layer_wide)
grad_u = tf.gradients(u, X_it)[0]
# partial_u_x = ((grad_u[0])[0])

partial_u_x = tf.gather(grad_u, [0], axis=-1)
partial_u_y = tf.gather(grad_u, [1], axis=-1)

# partial_u_x = tf.strided_slice(grad_u, [0], [5])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    x_it_batch = rand_it(batchsize_it, dim)
    fun_u, grad_result, grad_result_x, grad_result_y = sess.run([u, grad_u, partial_u_x, partial_u_y], feed_dict={X_it: x_it_batch})
    print(fun_u)
    print(grad_result)
    print(grad_result_x)
    print(grad_result_y)