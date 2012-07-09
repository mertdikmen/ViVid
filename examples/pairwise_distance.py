import pathfinder
import vivid
import numpy as np


A = np.random.random((500,1000)).astype('float32')
B = np.random.random((400,1000)).astype('float32')

dA = vivid.DeviceMatrix(A)
dB = vivid.DeviceMatrix(B)

dAcl = vivid.DeviceMatrixCL(A)
dBcl = vivid.DeviceMatrixCL(B)

# Pairwise distance using OpenCL
Ccl = vivid.pwdist_cl(dAcl, dBcl).mat()

# Pairwise distance using CUDA
Ccuda = vivid.pwdist_cuda(dA,dB).mat()

# Pairwise distance using Python (for reference)
reference = A[:,np.newaxis,:] - B[np.newaxis,:,:]
Cpython = (reference * reference).sum(axis=2)

if np.allclose(Ccl, Cpython, 1e-4):
    print("OpenCL OK")

if np.allclose(Ccuda, Cpython, 1e-4):
    print("CUDA OK")
