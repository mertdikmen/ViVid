#!/usr/bin/env /usr/bin/python2

import pathfinder
import vivid
import numpy as np
import unittest

from PIL import Image
from scipy.misc import fromimage

class TestImageReading(unittest.TestCase):
    def setUp(self):
        image_name = '../media/kewell1.jpg'
        
        self.reference_image_uint8 = fromimage(Image.open(image_name))

        self.fv = vivid.ImageSource(imlist=[image_name])
        self.cs = vivid.ConvertedSource(self.fv, target_type = vivid.cv.CV_32FC3)
        self.gs = vivid.GreySource(self.cs)

    def testRead(self):
        frame_uint8 = vivid.cvmat2array(self.fv.get_frame(0))

        assert(np.allclose(frame_uint8[:,:,::-1], self.reference_image_uint8))
        
if __name__ == '__main__':
    unittest.main()        
