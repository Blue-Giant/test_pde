import scipy.io
import numpy as np


# load the data from matlab of .mat
def loadMatlabIdata(filename=None):
    data = scipy.io.loadmat(filename, mat_dtype=True, struct_as_record=True)  # variable_names='CATC'
    return data


def test1():
    mat_file_name = 'data_a7.mat'
    mat_data = loadMatlabIdata(mat_file_name)
    mat_data_keys = mat_data.keys()  # 字典不支持数字索引mat_data_keys[1]，支持键值索引
    A = mat_data['A']
    f = mat_data['f']
    shapeA = np.shape(A)
    print(A)
    print(f)


def test2():
    # mat_file_name = '../data2Matlab/meshXY.mat'
    mat_file_name = '../data2Matlab' + str('/meshXY.mat')
    mat_data = loadMatlabIdata(mat_file_name)
    XY = mat_data['meshXY']
    print(mat_data)


#%% examples
if __name__ == '__main__':
    test2()