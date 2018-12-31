
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
	double duration = static_cast<double>(cv::getTickCount());
	//ss1_sc Data Generation
	{
		std::string File = "ss1_sc";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);

		std::string groundTruthFile = "ss3_scGroundTruth.csv";

		int GT_offset = 10;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss3_sc Data Generation
	{
		std::string File = "ss3_sc";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";
		
		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss1_sc_mb Data Generation
	{
		std::string File = "ss1_sc_mb";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";
		
		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss3_sc_mb Data Generation
	{
		std::string File = "ss3_sc_mb";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";
		
		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss1_mc Data Generation
	{
		std::string File = "ss1_mc";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";

		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss3_mc Data Generation
	{
		std::string File = "ss3_mc";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";

		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss1_mc_mb Data Generation
	{
		std::string File = "ss1_mc_mb";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";

		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	//ss3_mc_mb Data Generation
	{
		std::string File = "ss3_mc_mb";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_scGroundTruth.csv";

		int GT_offset = 3;
		DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	}

	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	duration /= 60;
	cout << "Total duration: " << duration << endl;
	//ss3_sc Data Evaluation
	//{
	//	//Read Ground truth from file
	//	vector<vector<int>> ss3_scGroundTruth;
	//	readGroundTruth(getFileString("ss3_scGroundTruth.csv"), ss3_scGroundTruth);

	//	//Read bounding rectangle data from file
	//	vector<vector<vector<vector<int>>>> boundRectDataSaliency;
	//	vector<vector<vector<vector<int>>>> boundRectDataGMM;

	//	readBoundRectData(getFileString("ss3_scBoundRectSaliency.csv"), boundRectDataSaliency);
	//	readBoundRectData(getFileString("ss3_scBoundRectGMM.csv"), boundRectDataGMM);

	//	vector<vector<int>> groundTruth = ss3_scGroundTruth;

	//	//Compare ground truth to found data - False/True positive rate
	//	vector<vector<int>> falsePositiveCountSaliency;
	//	vector<vector<int>> truePositiveCountSaliency;
	//	vector<vector<double>> precisionSaliency;
	//	vector<vector<double>> IoUSaliency;
	//	vector<double> recallSaliency;

	//	vector<vector<int>> falsePositiveCountGMM;
	//	vector<vector<int>> truePositiveCountGMM;
	//	vector<vector<double>> precisionGMM;
	//	vector<vector<double>> IoUGMM;
	//	vector<double> recallGMM;

	//	//Evaluation of Saliency detection results
	//	trueFalsePositiveRate(boundRectDataSaliency, groundTruth, falsePositiveCountSaliency, truePositiveCountSaliency, precisionSaliency, recallSaliency, IoUSaliency);
	//	trueFalsePositiveRate(boundRectDataGMM, groundTruth, falsePositiveCountGMM, truePositiveCountGMM, precisionGMM, recallGMM, IoUGMM);
	//}

	cout << "Done" << endl;
	_getch();
	return 0;
}




