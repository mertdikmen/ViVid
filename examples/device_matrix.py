import pathfinder
import vivid
import numpy as np

a = np.random.random((3,4)).astype('float32')

dm_a = vivid.DeviceMatrix(a)
