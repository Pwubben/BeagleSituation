
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
		cv::Mat src;
		std::stringstream ss;
		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		std::string file = "ss1_sc.avi";
		ss << path << file;
		std::string s = ss.str();
		cv::VideoCapture capture(s);
		for (int i = 0; i < 1200; i++) {
			capture >> src;
			imshow("src1", src);
		}
	}
	{
		cv::Mat src;
		std::stringstream ss;
		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		std::string file = "GroundTruth.avi";
		ss << path << file;
		std::string s = ss.str();
		cv::VideoCapture capture(s);
		for (int i = 0; i < 1200; i++) {
			capture >> src;
			imshow("src2", src);
		}
	}

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
		vector<vector<Rect>> boundRectSaliency;
		vector<Rect>	boundRectGMM;
		double avg_timeSaliency = 0.0, avg_timeGMM;

		//Performance parameters
		double max_dimension = 1200;
		double sample_step = 50;
		double stdThres = 5.0; // Mean+stdThres x stddev
		double backGroundRatio = 0.9;

		Mat data = cv::Mat::zeros(50, 3, CV_64F);
		int cycleCount = 0;
		vector<vector<vector<Rect>>> boundRectData;
		vector<double> avg_timeSaliencyData;

		//Run for all parameter combinations

		for (double max_dimension = 600; max_dimension < 801; max_dimension += 200) {
			for (double sample_step = 10; sample_step < 20; sample_step += 15) {
				for (double stdThres = 3.5; stdThres < 3.6; stdThres += 0.5) {
					capture.release();
					cv::VideoCapture capture(s);
					data.at<double>(cycleCount, 0) = max_dimension;
					data.at<double>(cycleCount, 1) = sample_step;
					data.at<double>(cycleCount, 2) = stdThres;
					
					SaliencyDetect(capture, boundRectSaliency, avg_timeSaliency, max_dimension, sample_step, stdThres);
					
					boundRectData.push_back(boundRectSaliency);
					avg_timeSaliencyData.push_back(avg_timeSaliency);
					cycleCount++;
				}
			}
		}
		
		//Evaluation

		for (int n = 0; n < boundRectData.size(); n++){
			for (int k = 0; k < boundRectData[n].size(); k++) {
				for (int l = 0; l < boundRectData[n][k].size(); l++) {

				}
			}
		}
	

		ofstream avgtimeFile("ss3_sc_avgTime.csv");

		int vsize = avg_timeSaliencyData.size();
		for (int n = 0; n<vsize; n++)
		{
			avgtimeFile << avg_timeSaliencyData[n] << endl;
		}


		ofstream vectorlabelFile("ss3_sc_vectorlabel.csv");
		vectorlabelFile << cv::format(data, cv::Formatter::FMT_CSV) << std::endl;
		vectorlabelFile.close();

		//GMMDetect(capture, boundRectGMM, avg_timeGMM, max_dimension, backGroundRatio);
	}

	//_getch();
	return 0;
}


