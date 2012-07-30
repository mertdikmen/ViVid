import pathfinder
import vivid
import numpy as np

a = np.random.random((3,4)).astype('float32')
dm_a = vivid.DeviceMatrix(a)
a_new = dm_a.mat()
assert(np.allclose(a, a_new))

dm_cl_a = vivid.DeviceMatrixCL(a)
assert(np.allclose(a, dm_cl_a.mat()))

a3d = np.random.random((3,4,5)).astype('float32')
dm_a3d = vivid.DeviceMatrix3D(a3d)
a3d_new = dm_a3d.mat()
assert(np.allclose(a3d, a3d_new))

dm_cl_a3d = vivid.DeviceMatrixCL3D(a3d)
assert(np.allclose(a3d, dm_cl_a3d.mat()))
