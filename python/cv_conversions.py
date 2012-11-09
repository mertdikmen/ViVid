import numpy as np
import cv

def cvmat2array(im):
    """
    Convert a CvMat to NumPy array
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

def dtype2depth(dtype, nChannels):
	if dtype == np.int8 and nChannels == 1:
		return cv.CV_8SC1
	if dtype == np.int8 and nChannels == 3:
		return cv.CV_8SC3
	if dtype == np.uint8 and nChannels == 1:
		return cv.CV_8UC1
	if dtype == np.uint8 and nChannels == 3:
		return cv.CV_8UC3
	if dtype == np.float32 and nChannels == 1:
		return cv.CV_32FC1
	if dtype == np.float32 and nChannels == 3:
		return cv.CV_32FC3
		
def array2cv(a):
    """
    Convert a NumPy array to CvMat
    """
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1

    cv_m = cv.CreateMat(a.shape[0], a.shape[1], dtype2depth(a.dtype, nChannels))

    cv.SetData(cv_m, a.tostring(), a.dtype.itemsize*nChannels*a.shape[1])

    return cv_m
