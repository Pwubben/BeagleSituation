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
	int  frameCount = 2;

	//Define rectangle on right side of screen
	cv::Rect testScr = cv::Rect(tSrc.cols - 50, tSrc.rows - 50, 40, 40);
	cv::cvtColor(tSrc, tSrc, CV_BGR2GRAY);
	frame1 = tSrc.clone();
	frame1 = frame1(testScr);
	compFrame = tSrc(testScr);
	bool isEqual = (cv::sum(frame1 != compFrame) == cv::Scalar(0, 0, 0, 0));
	//std::cout << isEqual << std::endl;
	bool eq;
	
	while ((cv::sum(frame1 != compFrame).val[0] < cv::Scalar(5).val[0])) {
		//cv::Scalar(0, 0, 0)
		//eq = cv::countNonZero(compFrame != frame1) == 0;
		std::cout << (cv::sum(frame1 != compFrame)) << std::endl;
	
		
		src >> tSrc;
		
		
		cv::cvtColor(tSrc, tSrc, CV_BGR2GRAY);
		compFrame = tSrc(testScr);
		frameCount++;
		//cv::imshow("Frame1", tSrc);
		//std::cout << frameCount << std::endl;
		//cv::waitKey(0);

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


void trim(cv::VideoCapture src, double begin, double end,int unPauseFrame, std::string name, int correction=0) {
	cv::Mat tSrc;
	src >> tSrc;
	
	int  count = 0;

	cv::VideoWriter video(name, CV_FOURCC('M', 'J', 'P', 'G'), 15, tSrc.size(), true);

	int startFrame = FPS*begin+correction;
	
	int endFrame = end- startFrame;
	endFrame = 1725;
	//endFrame = round(endFrame);
	// Cycle trough frames until video is unpaused

	for (int i = 0; i < unPauseFrame; i++) {
		src >> tSrc;
	}

	// Cycle trough frames until start frame is reached
	

	std:: cout << "Start frame reached, starting write" << std::endl;

	for (int i = 0; i < endFrame; i++) {
		//if (count == 0) {
		//	for (int j = 0; j < 1680; j++) {
		//		src >> tSrc;
		//		count++;
		//	}
		//}

		src >> tSrc;

		if (tSrc.empty())
		{
			// Reach end of the video file
			break;
		}
		video.write(tSrc);
		count++;
		/*cv::imshow("frame", tSrc);
		std::cout << count << std::endl;
		cv::waitKey(0);*/
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
	std::cout << count << std::endl;
	video.release();
}

double IMUData(std::string s, std::string d) {

	std::ifstream input(s);
	std::string line;
	std::stringstream lineparse;

	std::vector<double> ROTvec;
	std::vector<double> HDTvec;
	std::vector<double> lonvec;
	std::vector<double> latvec;
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
				number = std::stof(substr, &sz);
				ROTvec.push_back(number);
				//std::cout << "ROT entry: " << number << std::endl;
			}
			if (substr == "$GPGLL") {
				getline(lineparse, substr, ',');
				number = std::stof(substr, &sz);
				lonvec.push_back(number);
				getline(lineparse, substr, ',');
				if (substr == "S") {
					lonvec.back() *= -1;
					getline(lineparse, substr, ',');
					number = std::stof(substr, &sz);
					latvec.push_back(number);
					getline(lineparse, substr, ',');
					if (substr == "W") {
						latvec.back() *= -1;
					}
				}
				if (substr == "N") {
					getline(lineparse, substr, ',');
					number = std::stof(substr, &sz);
					latvec.push_back(number);
					getline(lineparse, substr, ',');
					if (substr == "W") {
						latvec.back() *= -1;
					}
				}
				//std::cout << "ROT entry: " << number << std::endl;
			}
		}
	}

	//float staticLat = float(-latvec[0]);

	int unPause = 0;
	while (latvec[unPause] == latvec[0])
	{
		unPause++;
	}

	int pause = latvec.size()-1;
	/*while (latvec[pause] == latvec[latvec.size()-1]) {
		pause--;
	}*/
	
	//unPause += 22;

	std::vector<double>::const_iterator first = ROTvec.begin() + unPause;
	std::vector<double>::const_iterator last = ROTvec.end() - (ROTvec.size() - pause);
	std::vector<double> ROTvec_ret1(first, last);
	first = HDTvec.begin() + unPause;
	last = HDTvec.end()- ( HDTvec.size() - pause);
	std::vector<double> HDTvec_ret1(first, last);
	first = latvec.begin() + unPause;
	last = latvec.end() -( latvec.size() - pause);
	std::vector<double> latvec_ret1(first, last);
	first = lonvec.begin() + unPause;
	last = lonvec.end()- ( lonvec.size() - pause);
	std::vector<double> lonvec_ret1(first, last);
	
	std::ofstream myfile(d, std::ofstream::out | std::ofstream::trunc);

	for (int i = 0; i < ROTvec_ret1.size(); i++) {
		myfile << latvec_ret1[i] << "," << lonvec_ret1[i] << "," << HDTvec_ret1[i] << "," << ROTvec_ret1[i] << std::endl;
	}
	myfile.close();
	double vidSize = lonvec.size();
	return vidSize;
}