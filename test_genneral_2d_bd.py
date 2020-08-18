import numpy as np
region_a = 0.0
region_b = 1.0
batch_size = 10
epsilon = 0.001
x_left_bd = np.tile([region_a, 0], (batch_size, 1))
x_right_bd = np.tile([region_b, 0], (batch_size, 1))
y_bottom_bd = np.tile([0, region_a], (batch_size, 1))
y_top_bd = np.tile([0, region_b], (batch_size, 1))
x_left_bd = x_left_bd.astype(np.float32)
x_right_bd = x_right_bd.astype(np.float32)
y_bottom_bd = y_bottom_bd.astype(np.float32)
y_top_bd = y_top_bd.astype(np.float32)
print('y_bottom_bd:\n', y_bottom_bd)
print('y_top_bd:\n', y_top_bd)

rand_num_array1 = (region_b - region_a) * np.random.random(batch_size) + region_a + epsilon
for ii in range(batch_size):
    x_left_bd[ii, 1] = rand_num_array1[ii]

rand_num_array2 = (region_b - region_a) * np.random.random(batch_size) + region_a + epsilon
for ii in range(batch_size):
    x_right_bd[ii, 1] = rand_num_array2[ii]

rand_num_array3 = (region_b - region_a) * np.random.random(batch_size) + region_a + epsilon
for ii in range(batch_size):
    y_bottom_bd[ii, 0] = rand_num_array3[ii]

rand_num_array4 = (region_b - region_a) * np.random.random(batch_size) + region_a + epsilon
for ii in range(batch_size):
    y_top_bd[ii, 0] = rand_num_array4[ii]

print(x_left_bd)
print(x_right_bd)
print('y_bottom_bd:\n', y_bottom_bd)
print('y_top_bd:\n', y_top_bd)