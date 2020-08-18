import tensorflow as tf
import numpy as np


# def moments(
#     x,
#     axes,
#     shift=None,  # pylint: disable=unused-argument
#     name=None,
#     keep_dims=False):

# 参数：
# x：一个tensor张量，即我们的输入数据
# axes：一个int型数组，它用来指定我们计算均值和方差的轴（这里不好理解，可以结合下面的例子）
# shift：当前实现中并没有用到
# name：用作计算moment操作的名称
# keep_dims：输出和输入是否保持相同的维度
#
# 返回：
# 两个tensor张量：均值和方差

def mean_var(input_variable):
    v_shape = input_variable.get_shape()
    axis = [len(v_shape) - 1]
    v_mean, v_var = tf.nn.moments(input_variable, axes=axis, keep_dims=True)
    return v_mean, v_var


# apply moving average for mean and var when train on batch
def mean_var_with_update(x_mean, x_var):
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    update_x_mean, update_x_var = ema.apply([x_mean, x_var])
    return update_x_mean, update_x_var


def my_batch_normalization(input_x):
    # Batch Normalize
    x_shape = input_x.get_shape()
    axis = [len(x_shape) - 1]
    x_mean, x_var = tf.nn.moments(input_x, axes=axis, keep_dims=True)
    # scale = tf.Variable(tf.ones([len(x_shape) - 1, 1]))
    # shift = tf.Variable(tf.zeros([len(x_shape) - 1, 1]))
    scale = tf.Variable(tf.ones([1]))   # 所有的batch 使用同一个scale因子
    shift = tf.Variable(tf.zeros([1]))  # 所有的batch 使用同一个shift项
    epsilon = 0.001

    # # apply moving average for mean and var when train on batch
    # ema = tf.train.ExponentialMovingAverage(decay=0.5)
    #
    # def mean_var_with_update():
    #     ema_apply_op = ema.apply([x_mean, x_var])
    #     with tf.control_dependencies([ema_apply_op]):
    #         return tf.identity(x_mean), tf.identity(x_var)

    # x_mean, x_var = mean_var_with_update()

    out_x = tf.nn.batch_normalization(input_x, x_mean, x_var, shift, scale, epsilon)
    return out_x


with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
    # x = tf.placeholder(tf.float32, [None, 3])
    x = tf.placeholder(tf.float32, [None, 3])
    x_mean, x_var = mean_var(x)
    y = my_batch_normalization(x)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # input_x = np.random.randn(3, 3)
    # input_x = [
    #     [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9]],
    #     [[1, 2, 3],
    #      [4, 5, 6],
    #      [7, 8, 9]]
    # ]

    input_x = [
        [1, 2, 3],
         [8, 5, 6],
         [7, 8, 18]
    ]
    m, v = sess.run([x_mean, x_var], feed_dict={x:input_x})
    # m, v = mean_var(input_x)
    print("均值：", m)
    print("方差：", v)

    # ma_mean, ma_var = mean_var_with_update(m, v)
    #
    # print("滑动均值：", ma_mean)
    # print("滑动方差：", ma_var)

    y_out = sess.run(y, feed_dict={x: input_x})
    print("归一化：", y_out)