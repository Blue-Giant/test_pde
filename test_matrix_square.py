import tensorflow as tf
import numpy as np

weight = tf.placeholder(tf.float32, name='weight', shape=[3, 3])
bias = tf.placeholder(tf.float32, name='weight', shape=[3, 1])
regular_w = 0
regular_b = 0
regular_w = regular_w + tf.reduce_sum(tf.square(weight), keep_dims=False)
regular_b = regular_b + tf.reduce_sum(tf.square(bias), keep_dims=False)

config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    W = np.array(
        [ [1,2,3],
          [4,5,6],
          [7,8,9]
        ])
    B = np.array(
        [[1],
         [2],
         [3]])
    w_norm2, b_norm2 = sess.run([regular_w, regular_b], feed_dict={weight:W, bias:B})

    print(w_norm2)
    print(b_norm2)