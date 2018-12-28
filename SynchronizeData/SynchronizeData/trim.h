#include "stdafx.h"
#include "opencv2/opencv.hpp"

void unPause(cv::VideoCapture src, int& begin);

void trim(cv::VideoCapture src, double begin, double end, int unPauseFrame, std::string name, int correction = 0);

void IMUData(std::string s, std::vector<double> & ROTvec_ret, std::vector<double> & HDTvec_ret);