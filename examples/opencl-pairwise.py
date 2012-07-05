import pathfinder
import vivid
import numpy as np

A = np.random.random((5,1000)).astype('float32')
B = np.random.random((4,1000)).astype('float32')

dA = vivid.DeviceMatrixCL(A)
dB = vivid.DeviceMatrixCL(B)

dC = vivid.pwdist_cl(dA,dB)

