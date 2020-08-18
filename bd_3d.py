import numpy as np
import tensorflow as tf


def rand_bd_2D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    epsilon = 0.0001
    if variable_dim == 2:
        left_bd = (region_b-region_a) * np.random.random([batch_size, 2]) + region_a + epsilon
        for ii in range(batch_size):
            left_bd[ii, 1] = region_a

        right_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a + epsilon
        for ii in range(batch_size):
            right_bd[ii, 1] = region_b

        bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a + epsilon
        for ii in range(batch_size):
            bottom_bd[ii, 0] = region_a

        top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a + epsilon
        for ii in range(batch_size):
            top_bd[ii, 0] = region_b

        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)

        return left_bd, right_bd, bottom_bd, top_bd
    else:
        return

region_b = 0.1
region_a = 0.0
batch_size = 10
epsilon = 0.0001
bottom_bd = (region_b - region_a) * np.random.random([batch_size, 3]) + region_a + epsilon
for ii in range(batch_size):
    bottom_bd[ii, 2] = 0.0

print(bottom_bd)

