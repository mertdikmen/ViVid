from cv_conversions import *

import numpy as np
import scipy.signal

def make_gauss_filter(M, stdev):
    gx = scipy.signal.gaussian(M, stdev, sym=True).reshape((1,-1))
    h = gx.T * gx

    hmag = np.sqrt(np.sum(h*h))

    return h / hmag

class SIFTSource:
    """
    Produce SIFT features on a regular grid
    """

    def __init__(self, origin):
        self.origin = origin

        gradient_filter_bw = 7.0
        weight_filter_bw = 16.0

    def get_sift(frame_num):
        raise NotImplementedError
