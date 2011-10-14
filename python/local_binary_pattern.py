from cv_conversions import *
from _vivid import compute_lbp_n8_r1_u2

class LocalBinaryPatternSource:
    r = 8
    def __init__(self, origin):
        self.origin = origin

    def get_lbp(self, frame_num):
        frame = cvmat2array(self.origin.get_frame(frame_num))
        res = compute_lbp_n8_r1_u2(frame)
        return res
