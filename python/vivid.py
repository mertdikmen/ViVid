import numpy as np
import scipy
import cv
import numpy as np

from collections import deque

#Supplementary functions
from vivid_kmeans import *

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

class CachedSource:
    """A simple cache.

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
def cvmat2array(im):
    """
    Converts a CvMat object to numpy array
    """
    depth2dtype = {
        cv.CV_8S: np.int8, cv.CV_8SC: np.int8, cv.CV_8SC1: np.int8, 
        cv.CV_8SC2: np.int8, cv.CV_8SC3: np.int8, cv.CV_8SC4: np.int8,
        cv.CV_8U: np.uint8, cv.CV_8UC: np.uint8, cv.CV_8UC1: np.uint8, 
        cv.CV_8UC2: np.uint8, cv.CV_8UC3: np.uint8, cv.CV_8UC4: np.uint8,

        cv.CV_16S: np.int16, cv.CV_16SC1: np.int16, cv.CV_16SC3: np.int16, 
        cv.CV_16U: np.uint16, cv.CV_16UC1: np.uint16, cv.CV_16UC3: np.uint16,
        cv.CV_16SC: np.int16, cv.CV_16SC2: np.int16, cv.CV_16SC4: np.int16, 
        cv.CV_16UC: np.uint16, cv.CV_16UC2: np.uint16, cv.CV_16UC4: np.uint16,

        cv.CV_32F: np.float32, cv.CV_32FC1: np.float32, cv.CV_32FC3: np.float32,
        cv.CV_32S: np.int32, cv.CV_32SC1: np.int32, cv.CV_32SC3: np.int32,
        cv.CV_32FC: np.float32, cv.CV_32FC2: np.float32, cv.CV_32FC4: np.float32,
        cv.CV_32SC: np.int32, cv.CV_32SC2: np.int32, cv.CV_32SC4: np.int32,

        cv.CV_64F: np.float64, cv.CV_64FC: np.float64, cv.CV_64FC1: np.float64,  
        cv.CV_64FC2: np.float64, cv.CV_64FC3: np.float64, cv.CV_64FC4: np.float64
    }

    arrdtype=im.channels

    a = np.fromstring(
            im.tostring(),
            dtype=depth2dtype[im.type],
            count=im.width*im.height*im.channels)

    if im.channels == 1:
        a.shape = (im.height, im.width)
    else:
        a.shape = (im.height,im.width,im.channels)

    return a

def array2cv(a):
    dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1

    cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
        dtype2depth[str(a.dtype)],
        channels)
    cv.SetData(cv_im, a.tostring(),
        a.dtype.itemsize*nChannels*a.shape[1])
    
    return cv_im
