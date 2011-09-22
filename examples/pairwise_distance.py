import pathfinder
import vivid
import numpy as np


A = np.random.random((5,1000)).astype('float32')
B = np.random.random((4,1000)).astype('float32')

dA = vivid.DeviceMatrix(A)
dB = vivid.DeviceMatrix(B)

# Pairwise distance using CUDA
C = vivid.pwdist_cuda(dA,dB).mat()

# Pairwise distance using Python (for reference)
reference = A[:,np.newaxis,:] - B[np.newaxis,:,:]
reference = (reference * reference).sum(axis=2)

print "Test: " + str(np.allclose(reference, C))
