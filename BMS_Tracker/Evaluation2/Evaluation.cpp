
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


	////ss3_sc_mb Data Generation
	{
		std::string File = "ss1_sc";
		std::string videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile;
		writeFileNames(File,videoFile, boundRectFileSal, avgTimeFileSal, boundRectFileGMM, avgTimeFileGMM, labelFile);
		std::string groundTruthFile = "ss1_scGroundtruth.csv";
		std::string ResultFile = "ss1_scResultFile.csv";

		int stopFrame = 1200;
		int GT_offset = 0;
		DataGeneration(videoFile, groundTruthFile, avgTimeFileSal, ResultFile, GT_offset, stopFrame);
	}


	cout << "Done" << endl;
	_getch();
	return 0;
}




