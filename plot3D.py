# coding:utf-8
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d
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


def true_solution_PDE1(input_dim=None, output_dim=None):
    u_true = lambda x, y: np.exp(-x)*(x + np.power(y, 3))
    return u_true


def true_solution_PDE2(input_dim=None, output_dim=None):
    u_true = lambda x, y: np.sin(np.pi * x)*(y**2)
    return u_true


def true_solution_PDE3(input_dim=None, output_dim=None):
    u_true = lambda x, y: np.exp(x)*np.exp(y)
    return u_true


def true_solution_PDE4(input_dim=None, output_dim=None):
    u_true = lambda x, y:  0.25*(np.power(x, 2)+np.power(y, 2))
    return u_true


def true_solution_PDE4_1(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.25 * (np.power(x, 2) + np.power(y, 2)) + x + y
    return u_true


def true_solution_PDE5(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.5 * (np.power(x, 2) * np.power(y, 2))
    return u_true


def true_solution_PDE6(input_dim=None, output_dim=None):
    u_true = lambda x, y: 0.5 * (np.power(x, 2) * np.power(y, 2)) + x + y
    return u_true


def true_solution_PDE7(input_dim=None, output_dim=None, eps=0.1):
    u_true = lambda x, y: x*y + eps**2/(4*(np.pi**2))*tf.sin(2*np.pi*x/eps)*tf.sin(2*np.pi*y/eps)
    return u_true


def true_solution_PDE8(input_dim=None, output_dim=None, eps=0.1):
    u_true = lambda x, y: 0.5*(x + y) + eps**2/(4*(np.pi**2))*tf.sin(2*np.pi*x/eps)*tf.sin(2*np.pi*y/eps)
    return u_true


def true_solution_PDE9(input_dim=None, output_dim=None, eps=0.1):
    u_true = lambda x, y: ((eps ** 2) / (4 * (np.pi ** 2))) * np.sin(2 * np.pi * (x / eps)) * np.sin(
        2 * np.pi * (y / eps)) + (1.0 / (4 * (np.pi ** 2))) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return u_true


def true_solution_PDE10(input_dim=None, output_dim=None, eps=0.1):
    u_true = lambda x, y: ((eps ** 2) / (4 * (np.pi ** 2))) * np.cos(2 * np.pi * (x / eps)) * np.cos(
        2 * np.pi * (y / eps)) + (1.0 / (4 * (np.pi ** 2))) * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return u_true


def true_solution_PDE11(input_dim=None, output_dim=None, eps=0.01):
    print('eps:', eps)
    u_true = lambda x, y: np.cos(2 * np.pi * (x / eps)) * np.cos(2 * np.pi * (y / eps)) + \
                          np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return u_true


def true_solution_PDE12(input_dim=None, output_dim=None, eps=0.01):
    print('eps:', eps)
    u_true = lambda x, y: np.cos(2 * np.pi * (x / eps)) * np.cos(2 * np.pi * (y / eps))
    return u_true


def true_solution_PDE13(input_dim=None, output_dim=None, eps=0.01):
    print('eps:', eps)
    u_true = lambda x, y: np.cos(2 * np.pi * (x / eps)) * np.cos(2 * np.pi * (y / eps)) + \
                          np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return u_true


def plot3d_1(R):
    region_l = 0.0
    region_r = 10.0
    test_bach_size = 100
    test_x = rand_it(test_bach_size, 1, region_l, region_r)
    test_y = rand_it(test_bach_size, 1, region_l, region_r)

    epsilon = R['epsilon']

    X, Y = np.meshgrid(test_x, test_y) # 格点矩阵,原来的x行向量向下复制len(y)次，形成len(y)*len(x)的矩阵，即为新的x矩阵；
    # 原来的y列向量向右复制len(x)次，形成len(y)*len(x)的矩阵，即为新的y矩阵；新的x矩阵和新的y矩阵shape相同
    if R['equa_name'] == 'PDE1':
        u_true = true_solution_PDE1()
    if R['equa_name'] == 'PDE2':
        u_true = true_solution_PDE2()
    if R['equa_name'] == 'PDE3':
        u_true = true_solution_PDE3()
    if R['equa_name'] == 'PDE4':
        u_true = true_solution_PDE4()
    if R['equa_name'] == 'PDE5':
        u_true = true_solution_PDE5()
    if R['equa_name'] == 'PDE6':
        u_true = true_solution_PDE6()
    if R['equa_name'] == 'PDE7':
        u_true = true_solution_PDE7(eps=epsilon)
    if R['equa_name'] == 'PDE8':
        u_true = true_solution_PDE8(eps=epsilon)
    if R['equa_name'] == 'PDE9':
        u_true = true_solution_PDE9(eps=epsilon)
    if R['equa_name'] == 'PDE10':
        u_true = true_solution_PDE10(eps=epsilon)
    if R['equa_name'] == 'PDE11':
        u_true = true_solution_PDE11(eps=epsilon)
    if R['equa_name'] == 'PDE12':
        u_true = true_solution_PDE12(eps=epsilon)
    if R['equa_name'] == 'PDE13':
        u_true = true_solution_PDE13(eps=epsilon)
    U_trur_test = u_true(X, Y)
    pltfig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, U_trur_test, c="r")
    plt.show()


def plt3d_2(R):
    # region_l = 0.0
    # region_r = 10.0
    # test_bach_size = 1000
    # test_x = rand_it(test_bach_size, 1, region_l, region_r)
    # test_y = rand_it(test_bach_size, 1, region_l, region_r)

    test_x = np.arange(0, 1, 0.01)
    test_y = np.arange(0, 1, 0.01)

    X, Y = np.meshgrid(test_x, test_y)  # 格点矩阵,原来的x行向量向下复制len(y)次，形成len(y)*len(x)的矩阵，即为新的x矩阵；
    # 原来的y列向量向右复制len(x)次，形成len(y)*len(x)的矩阵，即为新的y矩阵；新的x矩阵和新的y矩阵shape相同
    epsilon = R['epsilon']
    if R['equa_name'] == 'PDE1':
        u_true = true_solution_PDE1()
    if R['equa_name'] == 'PDE2':
        u_true = true_solution_PDE2()
    if R['equa_name'] == 'PDE3':
        u_true = true_solution_PDE3()
    if R['equa_name'] == 'PDE4':
        u_true = true_solution_PDE4()
    if R['equa_name'] == 'PDE5':
        u_true = true_solution_PDE5()
    if R['equa_name'] == 'PDE6':
        u_true = true_solution_PDE6()
    if R['equa_name'] == 'PDE7':
        u_true = true_solution_PDE7(eps=epsilon)
    if R['equa_name'] == 'PDE8':
        u_true = true_solution_PDE8(eps=epsilon)
    if R['equa_name'] == 'PDE9':
        u_true = true_solution_PDE9(eps=epsilon)
    if R['equa_name'] == 'PDE10':
        u_true = true_solution_PDE10(eps=epsilon)
    if R['equa_name'] == 'PDE11':
        u_true = true_solution_PDE11(eps=epsilon)
    if R['equa_name'] == 'PDE12':
        u_true = true_solution_PDE12(eps=epsilon)
    if R['equa_name'] == 'PDE13':
        u_true = true_solution_PDE13(eps=epsilon)
    U_trur_test = u_true(X, Y)

    fig = plt.figure(figsize=(8, 8))
    axes3 = Axes3D(fig)
    axes3.plot_surface(X, Y, U_trur_test, cmap=plt.get_cmap('rainbow'))
    plt.show()


if __name__ == "__main__":
    R = {}
    R['epsilon'] = 0.01
    # R['equa_name'] = 'PDE1'
    R['equa_name'] = 'PDE2'
    # R['equa_name'] = 'PDE3'
    # R['equa_name'] = 'PDE4'
    # R['equa_name'] = 'PDE4_1'
    # R['equa_name'] = 'PDE5'
    # R['equa_name'] = 'PDE6'
    # R['equa_name'] = 'PDE7'
    # R['equa_name'] = 'PDE8'
    # R['equa_name'] = 'PDE9'
    # R['equa_name'] = 'PDE10'
    # R['equa_name'] = 'PDE11'
    # R['equa_name'] = 'PDE12'
    # R['equa_name'] = 'PDE13'
    # plot3d_1()

    plt3d_2(R)