import numpy as np


def rand_it(size, variable_dim):
    x_it = np.random.rand(size, variable_dim)  # 通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    x_it = x_it.astype(np.float32)
    return x_it


def rand_bd(batch_size, variable_dim):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    if variable_dim == 1:
        x_bd = np.tile([0, 1], (batch_size, 1))
    else:
        # 下面这段的作用是什么？对于多维情形时的边界训练集设定！该如何理解
        # x_bd = np.random.randint(2, size=(batch_size, variable_dim))
        x_bd = np.tile([0, 1], (batch_size, 1))
    x_bd = x_bd.astype(np.float32)
    return x_bd


def rand_bd_2D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    if variable_dim == 2:
        x_left_bd = np.tile([region_a, 0], (batch_size, 1))
        x_right_bd = np.tile([region_b, 0], (batch_size, 1))
        y_bottom_bd = np.tile([0, region_a], (batch_size, 1))
        y_top_bd = np.tile([0, region_b], (batch_size, 1))
        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
        rand_num_array1 = (region_b-region_a) * np.random.random(batch_size) + region_a
        for ii in range(batch_size):
            x_left_bd[ii, 1] = rand_num_array1[ii]

        rand_num_array2 = (region_b - region_a) * np.random.random(batch_size) + region_a
        for ii in range(batch_size):
            x_right_bd[ii, 1] = rand_num_array2[ii]

        rand_num_array3 = (region_b - region_a) * np.random.random(batch_size) + region_a
        for ii in range(batch_size):
            y_bottom_bd[ii, 0] = rand_num_array3[ii]

        rand_num_array4 = (region_b - region_a) * np.random.random(batch_size) + region_a
        for ii in range(batch_size):
            y_top_bd[ii, 0] = rand_num_array4[ii]
        return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd
    else:
        return


size = 5
dim = 2
bd_x = rand_bd(size, dim)

X_it = rand_it(10, 1)
Y_it = rand_it(10, 1)
c = np.column_stack((X_it, Y_it))
print('dsfhdshfhsd')

b_size = 10
v_dim = 2
v_1 = 0
v_2 = 2
xl, xr, yb, yt = rand_bd_2D(b_size, v_dim, v_1, v_2)

xl

xr

yb

yt

exp1 = np.exp(-1)
print(exp1)