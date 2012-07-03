import pathfinder
import vivid
import numpy as np

A = np.random.random((5,1000)).astype('float32')
B = np.random.random((5,1000)).astype('float32')

dA = vivid.DeviceMatrixCL(A)

A_from = dA.mat()

assert(np.allclose(A, A_from))
