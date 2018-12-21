#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include "opencv2/opencv.hpp"

const int FPS = 15;

void trim(cv::VideoCapture src, double begin, double end) {
	cv::Mat tSrc;
	src >> tSrc;
	cv::VideoWriter video("Trimmedvid.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, tSrc.size(), true);

	int startFrame = FPS*begin;
	int endFrame = FPS*end - startFrame;

	// Cycle trough frames until start frame is reached
	for (int i = 0; i < startFrame; i++) {
		src >> tSrc;
	}

	for (int i = 0; i < endFrame; i++) {
		src >> tSrc;
		video.write(tSrc);
	}
	video.release();
}