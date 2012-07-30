import pathfinder
import vivid
import cPickle as pickle

import numpy as np
#import matplotlib.pyplot as plt

import time

# Create the reader for a list of image files
iv = vivid.ImageSource(
    imlist=['media/kewell1.jpg'])

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

ff = vivid.FlexibleFilter(
    gs,
    filter_bank=dictionary,
    optype=vivid.FF_OPTYPE_COSINE)

tic = time.time()
res_cuda = ff.filter_frame_cuda(0)
assignments_cuda = res_cuda[0]
weights_cuda = res_cuda[1]
toc = time.time()
print("FF CUDA total time: {0:.4f}".format(toc-tic))

tic = time.time()
res_cl = ff.filter_frame_cl(0)
assignments_cl = res_cl[0]
weights_cl = res_cl[1]
toc = time.time()
print("FF OPENCL total time: {0:.4f}".format(toc-tic))

res_c = ff.filter_frame_c(0)
assignments_c = res_c[0]
weights_c = res_c[1]
max_weight = weights_cuda.max()
max_weight_diff = np.abs(weights_cuda - weights_c).max()
print("Max weight diff with CUDA: {0:.9f}".format(float(max_weight_diff)))

max_weight_diff = np.abs(weights_cl - weights_c).max()
print("Max weight diff with OPENCL: {0:.9f}".format(float(max_weight_diff)))

#pickle.dump(res, open('pixel_assignments%04d.pkl'%dictionary_size,'w+'))
