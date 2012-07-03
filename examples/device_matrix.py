import pathfinder
import vivid
import numpy as np

a = np.random.random((3,4)).astype('float32')

dm_a = vivid.DeviceMatrix(a)

#copy back
a_new = dm_a.mat()

assert(np.allclose(a, a_new))

dm_cl_a = vivid.DeviceMatrixCL(a)
assert(np.allclose(a, dm_cl_a.mat()))
