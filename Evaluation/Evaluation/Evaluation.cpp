
#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include "lbp.hpp"
#include "opencv2/opencv.hpp"
#include "trim.h"
#include "BMS.h"
#include "RadarScreenDetect.h"
#include "DetectAlgorithms.h"

using namespace cv;
using namespace std;
int main()
{
	/// Video ss3_sc
	{
		// Declare VideoCapture object for storing video
		cv::Mat src, src_small;
		std::stringstream ss;
		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		std::string file = "ss1_sc.avi";
		ss << path << file;
		std::string s = ss.str();
		cv::VideoCapture capture(s);

		//Output parameters
		vector<Rect> boundRectSaliency, boundRectGMM;
		double avg_timeSaliency = 0.0, avg_timeGMM;
		
		//Performance parameters
		float max_dimension = 1200;
		int sample_step     =  25;
		double stdThres     = 5.0; // Mean+stdThres x stddev
		double backGroundRatio = 0.9;

		//SaliencyDetect(capture, boundRectSaliency, avg_timeSaliency, max_dimension, sample_step, stdThres);

		GMMDetect(capture, boundRectGMM, avg_timeGMM, max_dimension, backGroundRatio);
	}


    return 0;
}

