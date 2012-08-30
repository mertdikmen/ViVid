#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <fstream>
#include "magic.h"
#include <cv.h>
#include <opencv/cvaux.h>

#include <highgui.h>

#include <boost/python.hpp>

#define VIDEO_TYPE_AVI 1
#define VIDEO_TYPE_MPEG 0

class VideoReader{
    public:
        VideoReader(const std::string &videofilename);
        ~VideoReader();
        //CvImage get_frame(int framenum);
        boost::python::object get_frame(int framenum);
        int get_num_total_frames();

    private:
        int cur_framenum;
        int num_total_frames;
        cv::VideoCapture cap;
        std::string video_file_name;
        int video_type; //0: RAW-MPEG2, 1: AVI
};

