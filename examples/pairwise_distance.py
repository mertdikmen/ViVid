import pathfinder
import vivid
import numpy as np

import time

A = np.random.random((1000,1000)).astype('float32')
B = np.random.random((1000,1000)).astype('float32')

tic = time.time()
for i in range(10):
    Cc = vivid.pwdist_c(A,B)
toc = time.time()
print("C time (Python): {}".format(toc-tic))

# Pairwise distance using Python (for reference)
reference = A[:,np.newaxis,:] - B[np.newaxis,:,:]
Cpython = (reference * reference).sum(axis=2)

DO_OPENCL = False
DO_CUDA = True

if DO_OPENCL:
    dAcl = vivid.DeviceMatrixCL(A)
    dBcl = vivid.DeviceMatrixCL(B)

    # Pairwise distance using OpenCL
    tic = time.time()
    for i in range(10):
        Ccl = vivid.pwdist_cl(dAcl, dBcl).mat()
    toc = time.time()
    print("CL Time (Python): {}".format(toc - tic))

    if np.allclose(Ccl, Cpython, 1e-4):
        print("OpenCL Result OK")

if DO_CUDA:
    dA = vivid.DeviceMatrix(A)
    dB = vivid.DeviceMatrix(B)

    # Pairwise distance using CUDA
    tic = time.time()
    for i in range(10):
        Ccuda = vivid.pwdist_cuda(dA,dB).mat()
    toc = time.time()
    print("CUDA time (Python): {}".format(toc-tic))

    if np.allclose(Ccuda, Cpython, 1e-4):
        print("CUDA Result OK")
