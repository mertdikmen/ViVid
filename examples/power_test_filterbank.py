import pathfinder
import vivid

import numpy as np
from cv_conversions import cvmat2array

from scipy.misc import imresize

# Create the reader for a list of image files
iv = vivid.ImageSource(
    imlist=['media/kewell1.jpg'])

# Source for converting to float and scaling to [0,1]
cs = vivid.ConvertedSource(iv, vivid.cv.CV_32FC3, 1.0 / 255.0)

# Source for covnerting to grayscale 
gs = vivid.GreySource(cs)

frame = cvmat2array(gs.get_frame(0))

#PARAMETERS
NUM_FRAMES = 10
FRAME_HEIGHT = 200
FRAME_WIDTH = 200
dictionary_size = 100

frame = imresize(frame, (FRAME_HEIGHT, FRAME_WIDTH))
frames = [frame.copy() for i in range(NUM_FRAMES)]

word_size = 3
dictionary = np.random.random((dictionary_size, word_size, word_size))

ff = vivid.FlexibleFilter(
    gs,
    filter_bank=dictionary,
    optype=vivid.FF_OPTYPE_COSINE)

ff.filter_frame_cl_batch(frames)
