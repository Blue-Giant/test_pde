import numpy as np
import tensorflow as tf


def rand_bd_2D(batch_size, size2x=100, size2y=100, region_a=0, region_b=1, variable_dim=2):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    epsilon = 0.0001
    if variable_dim == 2:
        xy_bd = np.zeros(shape=[size2x, size2y])
        rand_num_array1 = (region_b - region_a) * np.random.random(size2x) + region_a + epsilon
        for ii in range(size2x):
            xy_bd[0, ii] = rand_num_array1[ii]

        rand_num_array2 = (region_b - region_a) * np.random.random(size2x) + region_a + epsilon
        for ii in range(size2x):
            xy_bd[size2y - 1, ii] = rand_num_array2[ii]

        rand_num_array3 = (region_b - region_a) * np.random.random(size2y) + region_a + epsilon
        for ii in range(size2y):
            xy_bd[ii, 0] = rand_num_array3[ii]

        rand_num_array4 = (region_b - region_a) * np.random.random(size2y) + region_a + epsilon
        for ii in range(size2y):
            xy_bd[ii, size2x - 1] = rand_num_array4[ii]
        xy_bd = np.expand_dims(xy_bd, axis=0)
        xy_bd = np.tile(xy_bd, [batch_size, 1, 1])
        return xy_bd
    else:
        return


if __name__ == '__main__':
    x_mesh_size = 10
    y_mesh_size = 10
    size2batch = 1
    xy = rand_bd_2D(batch_size=size2batch, size2x=x_mesh_size, size2y=y_mesh_size)
    print(xy)