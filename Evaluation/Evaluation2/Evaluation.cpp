
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
	

	//int i(2);
	//bool check(false);
	//while (i < 10) {
	//	try {
	//		if (!check) {
	//			//somefunction(i)
	//			i++;
	//		}
	//		else {
	//			check = false;
	//		}
	//	}
	//	catch (std::exception e) {
	//		check = true;
	//		std::cout << e.what() << std::endl;
	//	}
	//}



	//ss3_sc Data Generation
	//{
	//	std::string File = "ss3_sc";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
	//	std::string groundTruthFile = "ss3_scGroundTruth.csv";

	//	std::string SalResultFile = "ss3_scSalResultFile_MOG2add.csv";
	//	std::string GMMResultFile = "ss3_scGMMResultFile_MOG2add.csv";
	//	int GT_offset = 3;
	//	int stopFrame = 1250;
	//	DataGeneration(videoFile, groundTruthFile, SalResultFile, GMMResultFile, GMMResultFile, avgTimeFileGMM, labelFile, GT_offset,stopFrame);
	//}

	////ss1_sc Data Generation
	//{
	//	std::string File = "ss1_sc";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);

	//	std::string groundTruthFile = "ss3_scGroundTruth.csv";

	//	std::string SalResultFile = "ss1_scSalResultFile_MOG2add.csv";
	//	std::string GMMResultFile = "ss1_scGMMResultFile_MOG2add.csv";
	//	int GT_offset = 11;
	//	int stopFrame = 1250;
	//	DataGeneration(videoFile, groundTruthFile, SalResultFile, GMMResultFile, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset,stopFrame);
	//}
	////ss1_sc_mb Data Generation
	//{
	//	std::string File = "ss1_sc_mb";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
	//	std::string groundTruthFile = "ss1_sc_mbGroundTruth.csv";
	//	std::string GMMResultFile = "ss1_sc_mbGMMResultFile_MOG2add.csv";
	//	int stopFrame = 1200;
	//	int GT_offset = 0;
	//	DataGeneration(videoFile, groundTruthFile, avgTimeFileSal, GMMResultFile, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset,stopFrame);
	//}

	////ss3_sc_mb Data Generation
	{
		std::string File = "ss3_mc_mb";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss3_sc_mbGroundtruth.csv";
		std::string GMMResultFile = "ss3_sc_mbGMMResultFile_MOG2add.csv";

		int stopFrame = 1200;
		int GT_offset = 0;
		DataGeneration(videoFile, groundTruthFile, avgTimeFileSal, GMMResultFile, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset, stopFrame);
	}

	////ss1_mc Data Generation
	//{
	//	std::string File = "ss1_mc";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
	//	std::string groundTruthFile = "ss3_scGroundTruth.csv";

	//	int GT_offset = 3;
	//	DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	//}

	////ss3_mc Data Generation
	//{
	//	std::string File = "ss3_mc";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
	//	std::string groundTruthFile = "ss3_scGroundTruth.csv";

	//	int GT_offset = 3;
	//	DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	//}

	////ss1_mc_mb Data Generation
	//{
	//	std::string File = "ss1_mc_mb";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
	//	std::string groundTruthFile = "ss3_scGroundTruth.csv";

	//	int GT_offset = 3;
	//	DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	//}

	////ss3_mc_mb Data Generation
	//{
	//	std::string File = "ss3_mc_mb";
	//	std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
	//	writeFileNames(File, videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
	//	std::string groundTruthFile = "ss3_scGroundTruth.csv";

	//	int GT_offset = 3;
	//	DataGeneration(videoFile, groundTruthFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile, GT_offset);
	//}

	/*duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	duration /= 60;
	cout << "Total duration: " << duration << endl;*/
	//ss3_sc Data Evaluation
	//{
	//	//Read Ground truth from file
	//	vector<vector<int>> ss3_scGroundTruth;
	//	readGroundTruth(getFileString("ss3_scGroundTruth.csv"), ss3_scGroundTruth);

	//	//Read bounding rectangle data from file
	//	vector<vector<vector<vector<int>>>> boundRectDataSaliency;
	//	vector<vector<vector<vector<int>>>> boundRectDataGMM;

	//	readBoundRectData(getFileString("ss1_scBoundRectSaliency.csv"), boundRectDataSaliency);
	//	readBoundRectData(getFileString("ss1_scBoundRectGMM.csv"), boundRectDataGMM);

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
	//	trueFalsePositiveRate(boundRectDataSaliency, groundTruth, falsePositiveCountSaliency, truePositiveCountSaliency, precisionSaliency, recallSaliency, IoUSaliency,10);
	//	trueFalsePositiveRate(boundRectDataGMM, groundTruth, falsePositiveCountGMM, truePositiveCountGMM, precisionGMM, recallGMM, IoUGMM,10);

	//	ofstream FileSaliency(getFileString("ss1_scSaliencyResult.csv"), std::ofstream::out | std::ofstream::trunc);
	//	ofstream FileGMM(getFileString("ss1_scGMMResult.csv"), std::ofstream::out | std::ofstream::trunc);

	//	writeResultFile(falsePositiveCountSaliency, truePositiveCountSaliency, precisionSaliency, recallSaliency, IoUSaliency, FileSaliency);
	//	writeResultFile(falsePositiveCountGMM, truePositiveCountGMM, precisionGMM, recallGMM, IoUGMM, FileGMM);
	//}

	cout << "Done" << endl;
	_getch();
	return 0;
}




