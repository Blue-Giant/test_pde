import numpy as np
import tensorflow as tf


def weight_variable(shape, name2weight=None):
    if len(shape) == 4:
        xavier_stddev = np.sqrt(2 / (shape[2] + shape[3]))
    else:
        xavier_stddev = 0.1
    initial_V = tf.Variable(0.25*tf.truncated_normal(shape, stddev=xavier_stddev), dtype=tf.float32, name=name2weight)
    return initial_V


def construct_weights2kernel(height2kernel=2, width2kernel=3, hidden_layers=None,  in_channel=1, out_channel=2):
    layers = int(len(hidden_layers))

    kernels = []

    kernel = weight_variable([height2kernel, width2kernel, int(in_channel), hidden_layers[0]], name2weight='W_in')
    kernels.append(kernel)
    for i_layer in range(layers - 1):
        kernel = weight_variable([height2kernel, width2kernel, hidden_layers[i_layer], hidden_layers[i_layer + 1]], name2weight='W'+str(i_layer+1))
        kernels.append(kernel)

    kernel = weight_variable([height2kernel, width2kernel, hidden_layers[-1], int(out_channel)], name2weight='W_out')
    kernels.append(kernel)
    return kernels


def CNN_model(v_input, kernels=None, act_function=tf.nn.relu):
    # # 后面的1是channel的数量，因为我们输入的图片是黑白的，因此channel是1。如果是RGB图像，那么channel就是3.
    # out 的第一维是Batch_size大小
    dim2v = v_input.get_shape()
    if len(dim2v) == 2:
        out = tf.expand_dims(v_input, axis=-1)
        out = tf.expand_dims(out, axis=0)
    elif len(dim2v) == 3:
        out = tf.expand_dims(v_input, axis=-1)
    # out = tf.reshape(v_input, [-1, mesh_size, mesh_size, in_channel])
    # # print(x_image.shape) #[n_samples, 28,28,1]
    kernel = kernels[0]
    out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding="SAME")
    out = act_function(out)
    out = tf.reduce_mean(out, axis=0, keep_dims=True)

    len_weigths = len(kernels)
    for i_cnn in range(len_weigths-2):
        kernel = kernels[i_cnn+1]
        out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding="SAME")
        out = act_function(out)

    kernel_out = kernels[-1]
    out_result = tf.nn.conv2d(out, kernel_out, strides=[1, 1, 1, 1], padding="SAME")
    return out_result


def test_fun():
    hidden_layer = (2, 3, 4, 5, 6)
    batchsize_it = 2
    mesh_size = 10
    activate_func = tf.nn.relu
    weights2kernel = construct_weights2kernel(
        height2kernel=3, width2kernel=3, hidden_layers=hidden_layer, in_channel=1, out_channel=10)
    with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
        XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[batchsize_it, mesh_size, mesh_size])
        # XY_mesh = tf.placeholder(tf.float32, name='X_it', shape=[None, mesh_size, mesh_size])
        # 在变量的内部区域训练
        U_hat = CNN_model(XY_mesh, kernels=weights2kernel, act_function=activate_func)
        U_hat = tf.reduce_mean(U_hat, axis=-1)
        U_hat = tf.squeeze(U_hat)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        xy_mesh_batch = np.random.rand(batchsize_it, mesh_size, mesh_size)

        u = sess.run(U_hat, feed_dict={XY_mesh: xy_mesh_batch})
        print(u)


if __name__ == '__main__':
    test_fun()