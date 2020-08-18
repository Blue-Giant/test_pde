import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns


def hot1():
    # 产生10*10维矩阵
    a = np.random.uniform(0.5, 1.0, 100).reshape([10, 10])
    # 绘制热力图
    plt.imshow(a, interpolation='nearest', cmap=cm.coolwarm, origin='lower')
    plt.colorbar(shrink=.92)

    plt.xticks(())
    plt.yticks(())
    plt.show()


def hot2():
    x, y = np.random.rand(10), np.random.rand(10)
    z = (np.random.rand(9000000) + np.linspace(0, 1, 9000000)).reshape(3000, 3000)
    plt.imshow(z + 10, extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)), cmap=cm.hot, norm=LogNorm())
    plt.colorbar()
    plt.show()


def hot3():
    np.random.seed(20180316)
    x = np.random.randn(4, 4)
    f, ax1 = plt.subplots(figsize=(6, 6), nrows=1)
    sns.heatmap(x, annot=True, ax=ax1)
    plt.show()


if __name__ == '__main__':
    # hot2()
    # hot1()
    hot3()
