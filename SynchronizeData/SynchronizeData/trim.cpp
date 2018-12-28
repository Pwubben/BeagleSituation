#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include "opencv2/opencv.hpp"

const int FPS = 15;

void unPause(cv::VideoCapture src, int& begin) {
	cv::Mat tSrc, frame1,compFrame;
	src >> tSrc;
	src >> tSrc;
	int  frameCount = 0;

	//Define rectangle on right side of screen
	cv::Rect testScr = cv::Rect(tSrc.cols - 80, tSrc.rows - 80, 40, 40);
	cv::cvtColor(tSrc, tSrc, CV_BGR2GRAY);
	frame1 = tSrc.clone();
	frame1 = frame1(testScr);
	compFrame = tSrc(testScr);
	bool isEqual = (cv::sum(frame1 != compFrame) == cv::Scalar(0, 0, 0, 0));
	//std::cout << isEqual << std::endl;
	bool eq;
	
	while ((cv::sum(frame1 != compFrame) == cv::Scalar(0, 0, 0))) {
		//eq = cv::countNonZero(compFrame != frame1) == 0;
		
		/*cv::imshow("Frame", compFrame);
		cv::imshow("Frame1", frame1);*/
		
		src >> tSrc;
		cv::cvtColor(tSrc, tSrc, CV_BGR2GRAY);
		compFrame = tSrc(testScr);
		frameCount++;

		/*isEqual = (cv::sum(frame1 != compFrame) == cv::Scalar(0, 0, 0));
		std::cout << isEqual << std::endl;
		std::cout << "ret (python)  = " << std::endl << format(compFrame != frame1, cv::Formatter::FMT_PYTHON) << std::endl;
		cv::waitKey(0);*/

		if (cv::waitKey(30) > 0)
			break;
		
	}
	std::cout << frameCount << std::endl;
	begin = frameCount;
}

void trim(cv::VideoCapture src, double begin, double end,int unPauseFrame, std::string name, int correction) {
	cv::Mat tSrc;
	src >> tSrc;
	
	int  count = 0;

	cv::VideoWriter video(name, CV_FOURCC('M', 'J', 'P', 'G'), 15, tSrc.size(), true);

	int startFrame = FPS*begin+correction;
	int endFrame = FPS*end - startFrame;

	// Cycle trough frames until video is unpaused
	for (int i = 0; i < unPauseFrame; i++) {
		src >> tSrc;
	}

	// Cycle trough frames until start frame is reached
	for (int i = 0; i < startFrame; i++) {
		src >> tSrc;
	}

	std::cout << "Start frame reached, starting write" << std::endl;

	for (int i = 0; i < endFrame; i++) {
		src >> tSrc;

		if (tSrc.empty())
		{
			// Reach end of the video file
			break;
		}
		video.write(tSrc);
		count++;
		if (count == endFrame / 4) {
			std::cout << "25% done" << std::endl;
		}
		if (count == endFrame / 2) {
			std::cout << "50% done" << std::endl;
		}
		if (count == endFrame / 4*3) {
			std::cout << "75% done" << std::endl;
		}
	}
	std::cout << "Done" << std::endl;
	video.release();
}

void IMUData(std::string s, std::vector<double> & ROTvec_ret, std::vector<double> & HDTvec_ret) {

	std::ifstream input(s);
	std::string line;
	std::stringstream lineparse;

	std::vector<double> ROTvec;
	std::vector<double> HDTvec;
	std::string::size_type sz;

	while (std::getline(input, line, '\n')) {
		std::stringstream lineparse(line);
		while (lineparse.good()) {
			std::string substr;
			double number;
			getline(lineparse, substr, ',');
			if (substr == "$HEHDT") {
				getline(lineparse, substr, ',');
				number = std::stod(substr, &sz);
				HDTvec.push_back(number);
				//std::cout << "HDT entry: " << number << std::endl;
			}
			if (substr == "$HEROT") {
				getline(lineparse, substr, ',');
				number = std::stod(substr, &sz);
				ROTvec.push_back(number);
				//std::cout << "ROT entry: " << number << std::endl;
			}
		}
	}

	int unPause = 0;
	while (ROTvec[unPause] == 0)
	{
		unPause++;
	}
	unPause += 22;

	std::vector<double>::const_iterator first = ROTvec.begin() + unPause;
	std::vector<double>::const_iterator last = ROTvec.end();
	std::vector<double> ROTvec_ret1(first, last);
	first = HDTvec.begin() + unPause;
	last = HDTvec.end();
	std::vector<double> HDTvec_ret1(first, last);
	ROTvec_ret = ROTvec_ret1;
	HDTvec_ret = HDTvec_ret1;
}