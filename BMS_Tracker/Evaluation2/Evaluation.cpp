
#include "stdafx.h"
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <conio.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "BMS.h"
#include "Tracker.h"

using namespace cv;
using namespace std;

std::string getFileString(std::string fileName) {
	std::string path = "F:\\Afstuderen\\";
	std::stringstream ss;
	ss << path << fileName;
	std::string file = ss.str();
	return file;
}

int main()
{
	double duration = static_cast<double>(cv::getTickCount());

	//ss3_sc_mb Data Generation
	{
		std::string File = "ss1_sc.avi";
		std::string groundTruthFile = "ss1_scGroundtruth.csv";

		int stopFrame = 1200;
		int GT_offset = 0;

		Detection* detection = new Detection();
		detection->run(File, groundTruthFile, GT_offset, stopFrame);
	}


	cout << "Done" << endl;
	_getch();
	return 0;
}




