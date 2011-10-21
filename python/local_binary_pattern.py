from cv_conversions import *
from _vivid import compute_lbp_n8_r1
from _vivid import create_lbp_dictionary

class LocalBinaryPatternSource:
    # These parameters are fixed for the time being
    r = 1
    n = 8
    def __init__(self, origin, u=2):
        self.origin = origin
        self.u = u
        self.lbp_map = create_lbp_dictionary(u, self.n)

    def get_lbp(self, frame_num):
        frame = cvmat2array(self.origin.get_frame(frame_num))
        res = compute_lbp_n8_r1(frame, self.lbp_map)
        return res
