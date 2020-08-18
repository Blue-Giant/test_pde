import tensorflow as tf
import numpy as np


def test_gradient0():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x1 = tf.Variable(3, dtype=tf.float32, name='x_1')
            x2 = tf.Variable(1, dtype=tf.float32, name='x_2')
            y = 3 * (tf.pow(x1, 3)+tf.pow(x2, 3)) - 6*(tf.pow(x1, 2)+tf.pow(x2, 2))
            dx1, dx2 = tf.gradients(y, [x1, x2])  # 一阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx11, dx22 = sess.run([dx1, dx2], feed_dict={x1: 1, x2: 1})
        print('函数y在x_1=1, x_2=1处的关于x_1的一阶导数:dx1=', dx11, ',关于x_2的一阶导数:dx2=', dx22, sep='')


def test_gradient1():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x1 = tf.Variable(3, dtype=tf.float32, name='x_1')
            x2 = tf.Variable(1, dtype=tf.float32, name='x_2')
            y = 3 * (tf.pow(x1, 3)+tf.pow(x2, 3)) - 6*(tf.pow(x1, 2)+tf.pow(x2, 2)) + x1*x2
            dx1, dx2 = tf.gradients(y, [x1, x2])  # 一阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx11, dx22 = sess.run([dx1, dx2], feed_dict={x1: 1, x2: 1})
        print('函数y在x_1=1, x_2=1处的关于x_1的一阶导数:dx1=', dx11, ',关于x_2的一阶导数:dx2=', dx22, sep='')


def test_gradient2():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x1 = tf.Variable(3, dtype=tf.float32, name='x_1')
            x2 = tf.Variable(1, dtype=tf.float32, name='x_2')
            y = 3 * (tf.pow(x1, 3)+tf.pow(x2, 3)) - 6*(tf.pow(x1, 2)+tf.pow(x2, 2))
            dx1, dx2 = tf.gradients(y, [x1, x2])  # 一阶导数
            ddx1, ddx2 = tf.gradients([dx1, dx2], [x1, x2])  # 一阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ddx11, ddx22 = sess.run([ddx1, ddx2], feed_dict={x1: 1, x2: 1})
        print('函数y在x_1=1, x_2=1处的关于x_1的一阶导数:ddx1=', ddx11, ',关于x_2的一阶导数:ddx2=', ddx22, sep='')


def test_gradient3():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x1 = tf.Variable(3, dtype=tf.float32, name='x_1')
            x2 = tf.Variable(1, dtype=tf.float32, name='x_2')
            y = 3 * (tf.pow(x1, 3)+tf.pow(x2, 3)) - 6*(tf.pow(x1, 2)+tf.pow(x2, 2))+x*y
            dx1, dx2 = tf.gradients(y, [x1, x2])  # 一阶导数
            ddx1, ddx2 = tf.gradients([dx1, dx2], [x1, x2])  # 一阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ddx11, ddx22 = sess.run([ddx1, ddx2], feed_dict={x1: 1, x2: 1})
        print('函数y在x_1=1, x_2=1处的关于x_1的二阶导数:ddx1=', ddx11, ',关于x_2的二阶导数:ddx2=', ddx22, sep='')


def test_gradient4():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x1 = tf.Variable(3, dtype=tf.float32, name='x_1')
            x2 = tf.Variable(1, dtype=tf.float32, name='x_2')
            y = 3 * (tf.pow(x1, 3)+tf.pow(x2, 3)) - 6*(tf.pow(x1, 2)+tf.pow(x2, 2)) + (x1**2)*(x2**2)
            dx1, dx2 = tf.gradients(y, [x1, x2])  # 一阶导数
            ddx1, ddx2 = tf.gradients([dx1, dx2], [x1, x2])  # 一阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ddx11, ddx22 = sess.run([ddx1, ddx2], feed_dict={x1: 1, x2: 1})
        print('函数y在x_1=1, x_2=1处的关于x_1的二阶导数:ddx1=', ddx11, ',关于x_2的二阶导数:ddx2=', ddx22, sep='')


def test_gradient5():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.Variable([[0.1], [0.1]], dtype=tf.float32, name='x')
            y = 3*tf.pow(x, 3) - 6 * tf.pow(x, 2) + x
            dx = tf.gradients(y, x)  # 一阶导数
            ddx = tf.gradients(dx, x)  # 二阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx1, ddx1 = sess.run([dx, ddx], feed_dict={x: [[1], [1]]})
        print('函数y在x=[1;1]处的一阶导数:dx=', dx1, ',关于x的二阶导数:ddx=', ddx1, sep='')


def test_gradient6():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.Variable([[0.1], [0.1]], dtype=tf.float32, name='x')
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            B1 = np.array([[1], [3]], dtype=np.float32)
            W2 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            B2 = np.array([[2], [4]], dtype=np.float32)
            W3 = np.array([[2, 1]], dtype=np.float32)
            B3 = np.array([[2]], dtype=np.float32)
            y = tf.nn.relu(tf.matmul(W1, x) + B1)
            y = tf.nn.relu(tf.matmul(W2, y) + B2)
            y = tf.nn.relu(tf.matmul(W3, y) + B3)
            dx = tf.gradients(y, x)  # 一阶导数
            ddx = tf.gradients(dx, x)  # 二阶导数

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx1, ddx1 = sess.run([dx, ddx], feed_dict={x: [[1], [1]]})
        print('函数y在x=[1,1]处的一阶导数:dx=', dx1, ',关于x的二阶导数:ddx=', ddx1, sep='')


def test_gradient7():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.Variable([[0.1], [0.1]], dtype=tf.float32, name='x')
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            W2 = np.array([[1, 1]], dtype=np.float32)
            y = x*tf.matmul(W1, x)
            y = tf.matmul(W2, y)
            Dx = tf.gradients(y, x)[0]        # 一阶导数
            Dx1 = tf.gather(Dx, [0], axis=0)  # 将一阶导数关于x1的部分取出来
            Dx2 = tf.gather(Dx, [1], axis=0)  # 将一阶导数关于x2的部分取出来

            DDx1 = tf.gradients(Dx1, x)        # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            DDx1 = tf.squeeze(DDx1)
            DDx1 = tf.gather(DDx1, [0])  # 将一阶导数关于x2的部分取出来

            DDx2 = tf.gradients(Dx2, x)  # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            DDx2 = tf.squeeze(DDx2)
            DDx2 = tf.gather(DDx2, [1])  # 将一阶导数关于x1的部分取出来

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx, dx1, dx2, ddx1, ddx2 = sess.run([Dx, Dx1, Dx2, DDx1, DDx2], feed_dict={x: [[1], [1]]})
        print('dx:\n', dx)
        print('dx1:', dx1)
        print('dx2:', dx2)
        print('ddx1:', ddx1)
        print('ddx2:', ddx2)


def test_gradient8():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x1 = tf.Variable([[0.1]], dtype=tf.float32, name='x1')
            x2 = tf.Variable([[0.1]], dtype=tf.float32, name='x1')
            x = tf.concat([x1, x2], axis=0)
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            W2 = np.array([[1, 1]], dtype=np.float32)
            y = x*tf.matmul(W1, x)
            y = tf.matmul(W2, y)
            Dx = tf.gradients(y, x)[0]        # 一阶导数

            Dx1 = tf.gradients(y, x1)[0]
            DDx1 = tf.gradients(Dx1, x1)[0]       # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量

            Dx2 = tf.gradients(y, x2)[0]
            DDx2 = tf.gradients(Dx2, x2)[0]  # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx, dx1, dx2, ddx1, ddx2 = sess.run([Dx, Dx1, Dx2, DDx1, DDx2], feed_dict={x: [[1], [1]]})
        print('dx:\n', dx)
        print('dx1:', dx1)
        print('dx2:', dx2)
        print('ddx1:', ddx1)
        print('ddx2:', ddx2)


def test_gradient9():
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.placeholder(tf.float32, name='X_it', shape=[2, 3])
            # x = tf.Variable([[0.1, 0.1, 0,1], [0.1, 0.1, 0,1]], dtype=tf.float32, name='x')
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            W2 = np.array([[1, 1]], dtype=np.float32)
            y = x*tf.matmul(W1, x)
            y = tf.matmul(W2, y)
            Dx = tf.gradients(y, x)[0]        # 一阶导数
            Dx1 = tf.gather(Dx, [0], axis=0)  # 将一阶导数关于x1的部分取出来
            Dx2 = tf.gather(Dx, [1], axis=0)  # 将一阶导数关于x2的部分取出来

            DDx1 = tf.gradients(Dx1, x)        # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            DDx1 = tf.squeeze(DDx1)
            DDx1 = tf.gather(DDx1, [0], axis=0)  # 将一阶导数关于x2的部分取出来

            DDx2 = tf.gradients(Dx2, x)  # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            DDx2 = tf.squeeze(DDx2)
            DDx2 = tf.gather(DDx2, [1], axis=0)  # 将一阶导数关于x1的部分取出来

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx, dx1, dx2, ddx1, ddx2 = sess.run([Dx, Dx1, Dx2, DDx1, DDx2], feed_dict={x: [[1, 1, 1], [1, 1, 1]]})
        print('dx:\n', dx)
        print('dx1:\n', dx1)
        print('dx2:\n', dx2)
        print('ddx1:\n', ddx1)
        print('ddx2:\n', ddx2)


def test_gradient10(batch_size=3):
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.placeholder(tf.float32, name='X_it', shape=[None, 2])
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            W2 = np.array([[1], [1]], dtype=np.float32)
            y = x*tf.matmul(x, W1)
            y = tf.matmul(y, W2)
            Dx = tf.gradients(y, x)[0]        # 一阶导数
            Dx1 = tf.gather(Dx, [0], axis=-1)  # 将一阶导数关于x1的部分取出来
            Dx2 = tf.gather(Dx, [1], axis=-1)  # 将一阶导数关于x2的部分取出来

            DDx1 = tf.gradients(Dx1, x)[0]        # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            # DDx1 = tf.squeeze(DDx1)
            DDx1 = tf.gather(DDx1, [0], axis=-1)  # 将一阶导数关于x2的部分取出来

            DDx2 = tf.gradients(Dx2, x)[0]  # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            # DDx2 = tf.squeeze(DDx2)
            DDx2 = tf.gather(DDx2, [1], axis=-1)  # 将一阶导数关于x1的部分取出来

            TTT = -(DDx1+DDx2)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx, dx1, dx2, ddx1, ddx2, ttt= sess.run([Dx, Dx1, Dx2, DDx1, DDx2, TTT], feed_dict={x: [[1, 1], [1, 1], [1, 1]]})
        print('dx:\n', dx)
        print('dx1:\n', dx1)
        print('dx2:\n', dx2)
        print('ddx1:\n', ddx1)
        print('ddx2:\n', ddx2)
        print('ddx2:\n', ttt)


# 混合偏导数

def test_gradient11(batch_size=3):
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.placeholder(tf.float32, name='X_it', shape=[None, 2])
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            W2 = np.array([[1], [1]], dtype=np.float32)
            y = x*tf.matmul(x, W1)
            y = tf.matmul(y, W2)
            Dx = tf.gradients(y, x)[0]        # 一阶导数
            DDx = tf.gradients(Dx, x)[0]      # 二阶导数
            Dx1 = tf.gather(Dx, [0], axis=-1)  # 将一阶导数关于x1的部分取出来
            Dx2 = tf.gather(Dx, [1], axis=-1)  # 将一阶导数关于x2的部分取出来

            DDx1x2 = tf.gradients(Dx1, x)[0]        # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            DDx2x1 = tf.gradients(Dx2, x)[0]

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx, ddx, dx1, dx2, ddxy, ddyx= sess.run([Dx, DDx, Dx1, Dx2, DDx1x2, DDx2x1], feed_dict={x: [[1, 1], [1, 1], [1, 1]]})
        print('dx:\n', dx)
        print('ddx:\n', ddx)
        print('dx1:\n', dx1)
        print('dx2:\n', dx2)
        print('ddxy:\n', ddxy)
        print('ddyx:\n', ddyx)


# 混合偏导数
def test_gradient12(batch_size=3):
    with tf.device('/gpu:%s' % 0):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            # 注意类型必须均为浮点型，并且一致
            # 变量
            x = tf.placeholder(tf.float32, name='X_it', shape=[None, 2])
            W1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
            W2 = np.array([[1], [1]], dtype=np.float32)
            y = x*tf.matmul(x, W1)
            y = tf.matmul(y, W2)
            Dx = tf.gradients(y, x)[0]         # 一阶导数
            Dx1 = tf.gather(Dx, [0], axis=-1)  # 将一阶导数关于x1的部分取出来

            DDx1x2 = tf.gradients(Dx1, x)[0]   # 对取出来额部分关于x求导，就可以得到关于x的二阶导数的向量
            DDxy = tf.gather(DDx1x2, [1], axis=-1)  # 将一阶导数关于x2的部分取出来

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True  # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        dx, ddxy= sess.run([Dx, DDxy], feed_dict={x: [[1, 1], [1, 1], [1, 1]]})
        print('dx:\n', dx)
        print('ddxy:\n', ddxy)


if __name__ == "__main__":
    # test_gradient0()
    # test_gradient1()
    # test_gradient2()
    # test_gradient3()
    # test_gradient4()
    # test_gradient5()
    # test_gradient6()
    # test_gradient7()
    # test_gradient8()
    # test_gradient9()
    test_gradient10(batch_size=3)
    # test_gradient11(batch_size=3)
    # test_gradient12(batch_size=3)