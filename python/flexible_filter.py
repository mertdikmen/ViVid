import numpy as np

from _vivid import _update_filter_bank, DeviceMatrix, _filter_frame_cuda
from cv_conversions import *

##FLEXIBLE_FILTER CLASS
FF_OPTYPE_EUCLIDEAN = 0
FF_OPTYPE_COSINE = 1

class FlexibleFilter:
    """
    Applies a pre-specified operation in a convolution pattern on the image
    """
    def __init__(self, origin, filter_bank, optype=FF_OPTYPE_EUCLIDEAN):
        self.origin = origin
        self.optype = optype

        self.filter_bank = filter_bank.astype('float32')
        self.filter_bank_size = self.filter_bank.shape[0]
        self.filter_height = self.filter_bank.shape[1]
        self.filter_width =  self.filter_bank.shape[2]

        if len(self.filter_bank.shape) == 4:
            self.nchannels = self.filter_bank.shape[3]
        else:
            self.nchannels = 1
            self.filter_bank = self.filter_bank.reshape((self.filter_bank_size, 
                                                         self.filter_height, 
                                                         self.filter_width, 
                                                         self.nchannels))

        filter_bank_to_device = np.zeros((self.filter_bank_size, 
                                             self.filter_height, 
                                             self.filter_width * self.nchannels),dtype='float32')

        for chan_id in range(self.nchannels):
            filter_bank_to_device[:,:,chan_id::self.nchannels] = self.filter_bank[:,:,:,chan_id]

        _update_filter_bank(filter_bank_to_device)

        self.filter_half_height = int(self.filter_height / 2)
        self.filter_half_width  = int(self.filter_width / 2)

        assert((self.filter_height % 2 == 1) and
               (self.filter_width  % 2 == 1) )

    def filter_frame_cuda(self, framenum):
        frame = cvmat2array(self.origin.get_frame(framenum))

        if len(frame.shape) == 2:  #if this is a single channel image, fake the last channel
            frame = np.reshape(frame,(frame.shape[0], frame.shape[1], 1))

        #interleave the channels
        frame_ilv = np.empty((frame.shape[0], frame.shape[1] * frame.shape[2]),dtype='float32')
        for k in range(self.nchannels):
            frame_ilv[:,k::self.nchannels] = frame[:,:,k]

        frame_dm = DeviceMatrix(frame_ilv)

        result = _filter_frame_cuda(frame_dm,
                                  self.filter_bank_size,
                                  self.filter_height,
                                  self.filter_width,
                                  self.nchannels,
                                  self.optype)

        result = result.mat()

        apron_y = self.filter_height / 2
        apron_x = self.filter_width  / 2

        result[:, :apron_y,:] = -1
        result[:,-apron_y:,:] = -1
        result[:,:, :apron_x] = -1
        result[:,:,-apron_x:] = -1

        return result  
#
#    def filter_frame_cuda_noargmin(self, framenum):
#        frame = self.origin.get_frame(framenum)
#
#        if len(frame.shape) == 2:  #if this is a single channel image, fake the last channel
#            frame = numpy.reshape(frame,(frame.shape[0], frame.shape[1], 1))
#
#        #interleave the channels
#        frame_ilv = numpy.empty((frame.shape[0], frame.shape[1] * frame.shape[2]),dtype='float32')
#
#        for k in range(self.nchannels):
#            frame_ilv[:,k::self.nchannels] = frame[:,:,k]
#
#        frame_dm = DeviceMatrix(frame_ilv)
# 
#        return _filter_frame_cuda_noargmin(frame_dm,
#                                  self.filter_bank_size,
#                                  self.filter_height,
#                                  self.filter_width,
#                                  self.nchannels,
#                                  self.optype)
#
#
#    def cell_histogram_cuda(self, framenum, cell_size, offset_y, offset_x, n_bins):
#        print "flex filtering"
#        filter_result = self.filter_frame_cuda(framenum)
#        print "histogramming"
#        return _get_cell_histograms_cuda(filter_result, cell_size, offset_y, offset_x, n_bins)
#
#    def sum_square_difference(self, frame, filt):
#        frame_height = frame.shape[0]
#        frame_width = frame.shape[1]
#
#        dist = -numpy.ones((frame_height, frame_width),dtype='float32')
#
#        for i in range(self.filter_half_height, frame_height - self.filter_half_height):
#            for j in range(self.filter_half_width, frame_width - self.filter_half_width):
#                diff = frame[i-self.filter_half_height:i+self.filter_half_height + 1,
#                             j-self.filter_half_width :j+self.filter_half_width  + 1] - filt
#                dist[i,j] = numpy.sum(diff*diff)
#
#        return dist
#
#    def cosine_similarity(self, frame, filt):
#        frame_height = frame.shape[0]
#        frame_width = frame.shape[1]
#
#        dist = -numpy.ones((frame_height, frame_width),dtype='float32')
#        for i in range(self.filter_half_height, frame_height - self.filter_half_height):
#            for j in range(self.filter_half_width, frame_width - self.filter_half_width):
#                pw_mult = frame[i-self.filter_half_height:i+self.filter_half_height + 1,
#                                j-self.filter_half_width :j+self.filter_half_width  + 1,:] * filt
#                dist[i,j] = numpy.sum(pw_mult)
#
#        return dist
#
#    def filter_frame(self,framenum):
#        if self.optype == FF_OPTYPE_EUCLIDEAN:
#            atomic = self.sum_square_difference
#        elif self.optype == FF_OPTYPE_COSINE:
#            atomic = self.cosine_similarity
#
#        frame = self.origin.get_frame(framenum)
#        if hasattr(frame, 'mat'):
#            frame = frame.mat()
#        frame = frame.astype('float32')
#        frame_height = frame.shape[0]
#        frame_width  = frame.shape[1]
#
#        if self.nchannels == 1:
#            frame = frame.reshape((frame_height, frame_width, 1))
#
#        ids = -numpy.ones((frame_height, frame_width), dtype='int')
#        distances = -numpy.ones((frame_height, frame_width), dtype='float32')
#
##        cosine_filter_c(frame, self.filter_bank)
#
#        for k in range(self.filter_bank_size):
#            filter_res = atomic(frame, self.filter_bank[k])
#
#            if self.optype == FF_OPTYPE_EUCLIDEAN:
#                better_ids = filter_res < distances
#            elif self.optype == FF_OPTYPE_COSINE:
#                better_ids = filter_res > distances
#
#            distances[better_ids] = filter_res[better_ids]
#            ids[better_ids] = k
#
#        return ids, distances
#
#    def filter_frame_c(self,framenum):
#        if self.optype == FF_OPTYPE_EUCLIDEAN:
#            atomic = self.sum_square_difference
#        elif self.optype == FF_OPTYPE_COSINE:
#            atomic = self.cosine_similarity
#
#        frame = self.origin.get_frame(framenum)
#        if hasattr(frame, 'mat'):
#            frame = frame.mat()
#        frame = frame.astype('float32')
#        frame_height = frame.shape[0]
#        frame_width  = frame.shape[1]
#
#        if self.nchannels == 1:
#            frame = frame.reshape((frame_height, frame_width, 1))
#
#        ids = -numpy.ones((frame_height, frame_width), dtype='int')
#        distances = -numpy.ones((frame_height, frame_width), dtype='float32')
#
#        ret_mat =  cosine_filter_c(frame, self.filter_bank)
#
#        ids = ret_mat[:,:,0]
#        distances = ret_mat[:,:,1]    
#
#        return ids, distances


