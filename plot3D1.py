# coding:utf-8
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def rand_it(batch_size, variable_dim, region_a, region_b):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it

def true_solution_PDE(input_dim=None, output_dim=None):
    u_true = lambda x, y: np.exp(x)*np.exp(y)
    return u_true

region_l = 0.0
region_r = 1.0
test_bach_size = 100
test_x = rand_it(test_bach_size, 1, region_l, region_r)
test_y = rand_it(test_bach_size, 1, region_l, region_r)

X, Y = np.meshgrid(test_x, test_y)
u_true = true_solution_PDE()
U_trur_test = u_true(X, Y)
pltfig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X, Y, U_trur_test, c="r")
# ax.plot_surface(X, Y, U_trur_test)
# ax.plot_surface(X, Y, U_trur_test, rstride=1, cstride=1, cmap='rainbow')
plt.show()