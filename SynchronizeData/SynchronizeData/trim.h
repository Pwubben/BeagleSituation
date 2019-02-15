#include "stdafx.h"
#include "opencv2/opencv.hpp"

void unPause(cv::VideoCapture src, int& begin);

void Pause(cv::VideoCapture src, int& end);

void trim(cv::VideoCapture src, double begin, double end, int unPauseFrame, std::string name, int correction = 0);

double IMUData(std::string s, std::string d);