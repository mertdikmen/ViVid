import numpy as np
import scipy
import cv
import numpy as np

from collections import deque

#Supplementary functions
from vivid_kmeans import *
from flexible_filter import *
from local_binary_pattern import *
from cv_conversions import *

#This is the source for C++ implementations
from _vivid import *

#class FileVideo(object):
#    def __init__(self, file_name):
#        self.file_name = file_name
#        self.origin = cv.LoadVideo(file_name)
#
#    def get_frame(self, frame_num):
#        return self.origin.

class ImageSource:
    """
    Source form a collection of still images
    """
    def __init__(self, pattern="%.4d", imlist=None):
        """
        pattern - A string substution pattern to find the filename of
        a frame.  This will be called as "pattern % framenum"
        """
        self.pattern = pattern
        self.imlist = imlist

    def get_frame(self, frame_num):
        if self.imlist is None:
            image_file_name = self.pattern%frame_num
        else:
            image_file_name = self.imlist[frame_num]

        return cv.LoadImageM(image_file_name)

class ConvertedSource:
    """
    Change the datatype of the origin
    """
    def __init__(self, origin, target_type, scale=1.0):
        self.origin = origin
        self.target_type = target_type
        self.scale = scale

    def get_frame(self, frame_num):
        src = self.origin.get_frame(frame_num)
        ret = cv.CreateMat(src.rows, src.cols, self.target_type)
        if self.scale==1.0:
            cv.Convert(src,ret)
        else:
            cv.ConvertScale(src, ret, self.scale)
    
        return ret

class GreySource:
    """
    Convert the origin into grayscale
    """
    def __init__(self, origin):
        self.target_type = {
            cv.CV_8SC3: cv.CV_8SC1, 
            cv.CV_8UC3: cv.CV_8UC1,
            cv.CV_16SC3: cv.CV_16SC1,
            cv.CV_16UC3: cv.CV_16UC1,
            cv.CV_32FC3: cv.CV_32FC1,
            cv.CV_32SC3: cv.CV_32FC1,
            cv.CV_64FC3: cv.CV_64FC1
        }

        self.origin = origin

    def get_frame(self, frame_num):
        src = self.origin.get_frame(frame_num)
        ret = cv.CreateMat(src.rows, src.cols, self.target_type[src.type])
        cv.CvtColor(src, ret, cv.CV_BGR2GRAY)

        return ret

class ScaledSource:
    """
    Rescale the origin
    """
    def __init__(self, origin, scale, interpolation=cv.CV_INTER_LINEAR):
        self.scale = scale
        self.origin = origin
        self.interpolation = interpolation

    def get_frame(self, frame_num):
        src = self.origin.get_frame(frame_num)
        ret = cv.CreateMat(int(src.rows * self.scale), 
                           int(src.cols * self.scale), src.type)
        cv.Resize(src, ret, self.interpolation)

        return ret

class SquaredSource:
    """
    Returns the squared version of the origin
    """
    def __init__(self, origin):
        self.origin = origin

    def get_frame(self, frame_num):
        src = self.origin.get_frame(frame_num)
        cv.Mul(src,src,src)
        return src

class CachedSource:
    """
    A simple cache.

    Note: The cache eviction strategy is least recent insert, not LRU.

    """
    def __init__(self, origin, cache_size=30):
        self.origin = origin
        self.cache_size = cache_size
        self.cache = {}
        self.insert_order = deque()

    def get_frame(self, framenum):
        try:
            return self.cache[framenum]
        except KeyError:
            if len(self.cache) > self.cache_size-1:
                del self.cache[self.insert_order.popleft()]
            new = self.origin.get_frame(framenum)
            self.cache[framenum] = new
            self.insert_order.append(framenum)

        return self.cache[framenum]

# Helper functions

