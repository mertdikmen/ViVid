import pathfinder
import vivid
import cPickle as pickle

import numpy as np
#import matplotlib.pyplot as plt

import unittest

class TestFlexibleFilter(unittest.TestCase):
    def setUp(self):
        # Create the reader for a list of image files
        iv = vivid.ImageSource(
            imlist=['../media/kewell1.jpg'])
        
        # Source for converting to float and scaling to [0,1]
        cs = vivid.ConvertedSource(iv, vivid.cv.CV_32FC3, 1.0 / 255.0)
        
        # Source for covnerting to grayscale 
        gs = vivid.GreySource(cs)
        
        #dictionary = np.load('media/dictionary_300.npy')
        
        #word_size = 3
        #dictionary = dictionary.reshape((-1, word_size, word_size))
        
        dictionary_size = 1
        word_size = 3
        dictionary = np.ones((dictionary_size, word_size, word_size))
        #dictionary = np.random.random((dictionary_size, word_size, word_size))
        
        self.ff = vivid.FlexibleFilter(
            gs,
            filter_bank=dictionary,
            optype=vivid.FF_OPTYPE_COSINE)

        res_ref = self.ff.filter_frame_c(0)
        self.assignments_ref = res_ref[0]
        self.weights_ref = res_ref[1]
            
    def test_ffCUDA(self):
        res = self.ff.filter_frame_cuda(0)
        assignments = res[0]
        weights = res[1]
       
        self.assertTrue(np.allclose(assignments, self.assignments_ref))
        max_weight_diff = np.abs(weights - self.weights_ref).max()
        self.assertTrue(max_weight_diff < 10e-5)
        
    def test_ffOpenCL(self):
        res = self.ff.filter_frame_cl(0)
        assignments = res[0]
        weights = res[1]
       
        self.assertTrue(np.allclose(assignments, self.assignments_ref))
        max_weight_diff = np.abs(weights - self.weights_ref).max()
        self.assertTrue(max_weight_diff < 10e-5)