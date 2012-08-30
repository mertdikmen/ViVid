#include <stdexcept>
#include "VideoReader.hpp"
#include "NumPyWrapper.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL tb
#define NO_IMPORT
#include <numpy/arrayobject.h>

int VideoReader::get_num_total_frames(){
    int nFrames=0;
    if (video_type == VIDEO_TYPE_MPEG){
        std::cout << "Warning: There is no quick way of knowing the number of frames in MPEG2 format.  Returning -1" << std::endl;
        nFrames = num_total_frames;
    }
    else {
        char tempSize[4];
        std::ifstream videoFile(video_file_name.c_str(),std::ios::in|std::ios::binary);
        videoFile.seekg(0x30,std::ios::beg);
        videoFile.read(tempSize, 4);
        nFrames =           (unsigned char) tempSize[0] + 
                    0x100 * (unsigned char) tempSize[1] + 
                  0x10000 * (unsigned char) tempSize[2] + 
                0x1000000 * (unsigned char) tempSize[3];

        videoFile.close();
    }
    return nFrames;
}

VideoReader::VideoReader(const std::string &videofilename){
    cap = cv::VideoCapture(videofilename.c_str());
    video_file_name = videofilename;
    num_total_frames = -1;

    //Determine the video type
    magic_t myt = magic_open(MAGIC_NONE);
    magic_load(myt,NULL);
    std::string magic_output = magic_file(myt,videofilename.c_str());

    if (magic_output.find("AVI")!=std::string::npos){
        video_type = VIDEO_TYPE_AVI;
        num_total_frames = get_num_total_frames();
    }
    else if (magic_output.find("MPEG")!=std::string::npos){
        video_type = VIDEO_TYPE_MPEG;
    }
    else {
        std::cout << "ViVid: Cannot determine the video format.  Guessing MPEG2" << std::endl;
        video_type = VIDEO_TYPE_MPEG;
    }

    if (!cap.isOpened()){
        throw std::runtime_error("ViVid Error: Cannot initialize VideoCapture [OPENCV]");
    }

    magic_close(myt);

    cur_framenum = -1;
}

boost::python::object VideoReader::get_frame(int framenum){

    cv::Mat cv_frame;

    if ((num_total_frames != -1) && (framenum >= num_total_frames) ){
        throw std::out_of_range("ViVid: Frame index out of range");
    }

    if (framenum == cur_framenum){
        cap.retrieve(cv_frame);
    }

    else if ((framenum < cur_framenum + 20) && (framenum > cur_framenum)) {
        while (cur_framenum < framenum){
            cap.grab();
            cur_framenum++;
        }

        cap.retrieve(cv_frame);
    }
    else {
        if (video_type==1){  //AVI SEEKING HACK: Don't seek just iterate till the correct frame
            cap.set(CV_CAP_PROP_POS_FRAMES, 0.0);
            for (int framecount=0; framecount < framenum; framecount++)
                cap.grab();
            cap.retrieve(cv_frame);

        }
        else { //MPEG2
            int framemult = 3600;

            double targetnum = 48599.0 + (framenum * framemult);

            cap.set(CV_CAP_PROP_POS_FRAMES, double(framenum-170)*framemult + 48599);
            cap.grab();

            while (cap.get(CV_CAP_PROP_POS_FRAMES) < targetnum){
                cap.grab();
            }

            cap.retrieve(cv_frame);
        }
        cur_framenum = framenum;
    }

    const int height = cv_frame.size[0];
    const int width = cv_frame.size[1];

    PyObject* arr;

    cv::Mat float_frame;
    if (cv_frame.channels() == 1)
    {
        //cv_frame.convertTo(float_frame, CV_32SC1);
        npy_intp dims[2] = {height, width};
        arr = PyArray_New(&PyArray_Type, 2, dims,
                PyArray_UINT8, NULL, NULL, 0, 0, NULL);
        memcpy(PyArray_DATA(arr), cv_frame.data,
               height * width);

    }
    else if (cv_frame.channels() == 3)
    {
        npy_intp dims[3] = {height, width, 3};
        //cv_frame.convertTo(float_frame, CV_32SC3);
        arr = PyArray_New(&PyArray_Type, 3, dims,
                PyArray_UINT8, NULL, NULL, 0, 0, NULL);

        memcpy(PyArray_DATA(arr), cv_frame.data,
               height * width * 3);
    }
    else 
    {
        std::cerr << "ViVid: Cannot handle number of channels: " <<
            cv_frame.channels() << std::endl;
    }


    boost::python::handle<> temp(arr);
    boost::python::object retval(temp);

    return retval;
}

VideoReader::~VideoReader()
{
    //cap.release();
}

