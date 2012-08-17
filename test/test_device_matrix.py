import pathfinder
import vivid
import numpy as np
import unittest


class TestDeviceMatrix(unittest.TestCase):
    def setUp(self):
        self.A = np.random.random((400,500)).astype('float32')
        self.B = np.random.random((400,500,3)).astype('float32')
        
    def test_DeviceMatrixCUDA(self):
        dm = vivid.DeviceMatrix(self.A)
        dm_back = dm.mat()
        
        self.assertTrue(np.allclose(dm_back, self.A))

    def test_DeviceMatrixOpenCL(self):
        dm = vivid.DeviceMatrixCL(self.A)
        dm_back = dm.mat()
        
        self.assertTrue(np.allclose(dm_back, self.A))
        
    def test_DeviceMatrix3DCUDA(self):
        dm = vivid.DeviceMatrix3D(self.B)
        dm_back = dm.mat()
        
        self.assertTrue(np.allclose(dm_back, self.B))
        
    def test_DeviceMatrix3DOpenCL(self):
        dm = vivid.DeviceMatrixCL3D(self.B)
        dm_back = dm.mat()
        
        self.assertTrue(np.allclose(dm_back, self.B))       
        
if __name__ == '__main__':
    unittest.main()