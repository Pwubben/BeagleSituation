
#include "stdafx.h"
#include <iostream>
#include <stdio.h>
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
		std::string file = getFileString("ss3_sc.avi");

		// Declare VideoCapture object for storing video
		cv::VideoCapture capture(file);

		//Output parameters of lgorithms
		vector<vector<Rect>> boundRectSaliency, boundRectGMM;
		double avg_timeSaliency = 0.0, avg_timeGMM = 0.0;

		//Output storage vectors over multiple parameter settings
		vector<vector<vector<Rect>>> boundRectSaliencyData, boundRectGMMData;
		vector<double> avg_timeSaliencyData, avg_timeGMMData;

		//Matrix for storing parameter combinations
		Mat data = cv::Mat::zeros(50, 5, CV_64F);
		int cycleCountSal = 0;
		int cycleCountGMM = 0;

		//Run for all desired parameter combinations

		for (double max_dimension = 800; max_dimension < 801; max_dimension += 200) {
			for (double sample_step = 10; sample_step < 26; sample_step += 15) {
				for (double stdThres = 3.5; stdThres < 3.6; stdThres += 0.5) {
					//Clear data
					boundRectSaliency.clear();
					capture.release();

					cv::VideoCapture capture(file);
					//Parameter documentation
					data.at<double>(cycleCountSal, 0) = max_dimension;
					data.at<double>(cycleCountSal, 1) = sample_step;
					data.at<double>(cycleCountSal, 2) = stdThres;
					
					SaliencyDetect(capture, boundRectSaliency, avg_timeSaliency, max_dimension, sample_step, stdThres);
					
					boundRectSaliencyData.push_back(boundRectSaliency);
					avg_timeSaliencyData.push_back(avg_timeSaliency);
					cycleCountSal++;
				}
			}
			for (double backGroundRatio = 0.65; backGroundRatio < 0.96; backGroundRatio += 0.15) {
				
				//Clear data
				boundRectGMM.clear();
				capture.release();

				cv::VideoCapture capture(file);

				data.at<double>(cycleCountGMM, 3) = max_dimension;
				data.at<double>(cycleCountGMM, 4) = backGroundRatio;

				GMMDetect(capture, boundRectGMM, avg_timeGMM, max_dimension, backGroundRatio);
				boundRectGMMData.push_back(boundRectGMM);
				avg_timeGMMData.push_back(avg_timeGMM);

				cycleCountGMM++;
			}
		}
		
		//Saliency
		//Bounding rect vector file write
		file = getFileString("ss3_scBoundRectSaliency.csv");
		std::ofstream ss3_scBoundRectSalFile(file);

		writeBoundRectFile(boundRectSaliencyData, ss3_scBoundRectSalFile);

		//Average time file write
		file = getFileString("ss3_sc_avgTimeSaliency.csv");
		remove(file);
		ofstream avgtimeSalFile(file);

		for (int n = 0; n<avg_timeSaliencyData.size(); n++)
		{
			avgtimeSalFile << avg_timeSaliencyData[n] << endl;
		}
		avgtimeSalFile.close();

		//GMM
		//Bounding rect vector file write
		file = getFileString("ss3_scBoundRectGMM.csv");
		std::ofstream ss3_scBoundRectGMMFile(file);

		writeBoundRectFile(boundRectGMMData, ss3_scBoundRectGMMFile);

		//Average time file write
		file = getFileString("ss3_sc_avgGMM.csv");

		ofstream avgtimeGMMFile(file);
		for (int n = 0; n<avg_timeSaliencyData.size(); n++)
		{
			avgtimeGMMFile << avg_timeSaliencyData[n] << endl;
		}
		avgtimeGMMFile.close();

		//Parameter file write
		file = getFileString("ss3_sc_vectorlabel.csv");
		ofstream vectorlabelFile(file);

		vectorlabelFile << cv::format(data, cv::Formatter::FMT_CSV) << std::endl;
		vectorlabelFile.close();

		//GMMDetect(capture, boundRectGMM, avg_timeGMM, max_dimension, backGroundRatio);
	}

	//ss3_sc Data Evaluation
	{
		//Read Ground truth from file
		vector<vector<int>> ss3_scGroundTruth;
		readGroundTruth(getFileString("ss3_scGroundTruth.csv"), ss3_scGroundTruth);

		//Read bounding rectangle data from file
		vector<vector<vector<vector<int>>>> boundRectDataSaliency;
		vector<vector<vector<vector<int>>>> boundRectDataGMM;

		readBoundRectData(getFileString("ss3_scBoundRectSaliency.csv"), boundRectDataSaliency);
		readBoundRectData(getFileString("ss3_scBoundRectGMM.csv"), boundRectDataGMM);

		vector<vector<int>> groundTruth = ss3_scGroundTruth;

		//Compare ground truth to found data - False/True positive rate
		vector<vector<int>> falsePositiveCountSaliency;
		vector<vector<int>> truePositiveCountSaliency;
		vector<vector<double>> precisionSaliency;
		vector<vector<double>> IoUSaliency;
			   vector<double> recallSaliency;

		vector<vector<int>> falsePositiveCountGMM;
		vector<vector<int>> truePositiveCountGMM;
		vector<vector<double>> precisionGMM;
		vector<vector<double>> IoUGMM;
			   vector<double> recallGMM;

		//Evaluation of Saliency detection results
		trueFalsePositiveRate(boundRectDataSaliency, groundTruth, falsePositiveCountSaliency, truePositiveCountSaliency, precisionSaliency, recallSaliency, IoUSaliency);
		trueFalsePositiveRate(boundRectDataGMM, groundTruth, falsePositiveCountGMM, truePositiveCountGMM, precisionGMM, recallGMM, IoUGMM);


	}
	_getch();
	return 0;
}




