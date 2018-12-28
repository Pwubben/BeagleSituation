
#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include <vector>
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


	//ss3_sc Data Generation
	{

		// Declare VideoCapture object for storing video
		std::string file = "F:\\Afstuderen\\ss1_sc.avi";
		cv::VideoCapture capture(file);

		//Output parameters of lgorithms
		vector<vector<Rect>> boundRectSaliency;
		vector<Rect> boundRectGMM;
		double avg_timeSaliency = 0.0, avg_timeGMM;

		//Output storage vectors over multiple parameter settings
		vector<vector<vector<Rect>>> boundRectData;
		vector<double> avg_timeSaliencyData;

		//Performance parameters
		double max_dimension   = 1200;
		double sample_step     = 50;
		double stdThres        = 5.0;  // Mean+stdThres x stddev
		double backGroundRatio = 0.9;

		//Matrix for storing parameter combinations
		Mat data = cv::Mat::zeros(50, 3, CV_64F);
		int cycleCount = 0;

		//Run for all desired parameter combinations

		for (double max_dimension = 600; max_dimension < 801; max_dimension += 200) {
			for (double sample_step = 10; sample_step < 20; sample_step += 15) {
				for (double stdThres = 3.5; stdThres < 3.6; stdThres += 0.5) {
					//Clear data
					boundRectSaliency.clear();
					capture.release();

					cv::VideoCapture capture(file);
					//Parameter documentation
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
		//Bounding rect vector file write
		std::ofstream ss3_scBoundRectFile("F:\\Afstuderen\\ss3_scBoundRect.csv");
		//Parameter 
		for (int n = 0; n < boundRectData.size(); n++){
			//Time 
			for (int k = 0; k < boundRectData[n].size(); k++) {
				//Bounding box 
				for (int l = 0; l < boundRectData[n][k].size(); l++) {
					ss3_scBoundRectFile << boundRectData[n][k][l].x << "," << boundRectData[n][k][l].y << "," << boundRectData[n][k][l].width << "," << boundRectData[n][k][l].height << endl;
				}
				ss3_scBoundRectFile << "NT" << endl;
			}
			ss3_scBoundRectFile << "NP" << endl;
		}
		ss3_scBoundRectFile.close();

		//Average time file write
		ofstream avgtimeFile("ss3_sc_avgTime.csv");
		for (int n = 0; n<avg_timeSaliencyData.size(); n++)
		{
			avgtimeFile << avg_timeSaliencyData[n] << endl;
		}
		
		//Parameter file write
		ofstream vectorlabelFile("ss3_sc_vectorlabel.csv");
		vectorlabelFile << cv::format(data, cv::Formatter::FMT_CSV) << std::endl;
		vectorlabelFile.close();

		//GMMDetect(capture, boundRectGMM, avg_timeGMM, max_dimension, backGroundRatio);
	}

	//ss3_sc Data Evaluation
	{
		//Read Ground truth from file
		vector<vector<int>> ss3_scGroundTruth;
		readGroundTruth("F:\\Afstuderen\\ss3_scGroundTruth.csv", ss3_scGroundTruth);

		//Read bounding rectangle data from file
		vector<vector<vector<vector<int>>>> boundRectData;
		readBoundRectData("F:\\Afstuderen\\ss3_scBoundRect.csv", boundRectData);

		vector<vector<int>> groundTruth = ss3_scGroundTruth;

		//Compare ground truth to found data
		int falsePositiveCount = 0;
		int truePositiveCount = 0;

		//Parameter
		for (int i = 0; i < boundRectData.size(); i++) {
			//Time
			for (int j = 0; j < boundRectData[i].size(); j++) {
				cv::Point l2(groundTruth[j][0], groundTruth[j][1]);
				cv::Point r2(groundTruth[j][0] + groundTruth[j][2], groundTruth[j][1] + groundTruth[j][3]);

				//Bounding box
				for (int k = 0; k < boundRectData[i][j].size(); k++) {
					cv::Point l1(boundRectData[i][j][k][0], boundRectData[i][j][k][1]);
					cv::Point r1(boundRectData[i][j][k][0]+ boundRectData[i][j][k][2], boundRectData[i][j][k][1]+ boundRectData[i][j][k][3]);
					if (doOverlap(l1, r1, l2, r2)) {
						truePositiveCount++;
					}
					else
						falsePositiveCount++;

				}
			}
		}
	}
	_getch();
	return 0;
}




