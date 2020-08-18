import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import my_prob
import PDE_tools
from PDE_CNN import PDE_CNN1d_base


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    # PDE_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_units']), log_fileout)
    # PDE_tools.log_string('activate function:%s\n' % str(R_dic['act_name']), log_fileout)

    PDE_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)

    PDE_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)

    if R_dic['epsilon'] != 0:
        PDE_tools.log_string('epsilon: %f\n' % (R_dic['epsilon']), log_fileout)  # 替换上两行

    if R_dic['variational_loss'] == 1:
        PDE_tools.log_string('Loss function: variational loss\n', log_fileout)
    else:
        PDE_tools.log_string('Loss function: original function loss\n', log_fileout)

    PDE_tools.log_string('hidden layer:%s\n' % str(R_dic['hidden_layers']), log_fileout)

    if (R_dic['optimizer_name']).title() == 'Adam':
        PDE_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        PDE_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    if R_dic['activate_stop'] != 0:
        PDE_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        PDE_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


def xavier_init(in_dim, out_dim, weight_name='weight'):
    xavier_stddev = np.sqrt(2/(in_dim + out_dim))
    Var = tf.Variable(tf.truncated_normal([1, in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32, name=weight_name)
    return Var


#  区域[a,b]生成随机数, n代表变量个数
def rand_1d_region_column(batch_size, input_dim, region_a=0, region_b=1):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, input_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


def rand_bd_1D_column(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 1:
        x_left_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_a
        x_right_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_b
        return x_left_bd, x_right_bd
    else:
        return


def solve_PDE(R):
    batch_size = 500

    input_dim = 1
    out_dim = 1
    hiddens = [10, 20, 30]

    # 问题区域，每个方向设置为一样的长度
    region_lb = 0.0
    region_rt = 1.0
    size2step = 2

    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            X_it = tf.placeholder(tf.float32, name='X_it', shape=[batch_size, 1])
            # 在变量的内部区域训练
            W = xavier_init(input_dim, hiddens[0], weight_name='W-transInput' + str(hiddens[0]))
            B = tf.Variable(0.1 * tf.random.uniform([1, int(batch_size/2), hiddens[0]]), dtype='float32',
                            name='B-transInput' + str(hiddens[0]))
            output = PDE_CNN1d_base.my_conv1d_column(X_it, W, B, step_size=size2step)

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        x_it_batch = rand_1d_region_column(batch_size, input_dim, region_a=region_lb, region_b=region_rt)
        u__hat = sess.run(output, feed_dict={X_it: x_it_batch})
        print(u__hat)


if __name__ == '__main__':
    R = {}
    R['gpuNo'] = 0
    solve_PDE(R)
