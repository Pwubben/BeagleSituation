#include <iostream>
#include "GnuGraph.h"
#include <stdio.h>
#include <ctime>
#include <conio.h>
#include <vector>
#include "opencv2/opencv.hpp"
#include "BMS.h"
#include "Tracker.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	
	//waitKey(0);
	double duration = static_cast<double>(cv::getTickCount());

	//ss3_sc_mb Data Generation
	{
		std::string File = "SS5_1T_R926.avi";
		std::string groundTruthFile = "ss1_scGroundtruth.csv";
		std::string beagleFile = "SS5_1T_R926_Beagle.csv";

		Detection* detection = new Detection();
		detection->run(File, groundTruthFile,beagleFile);
	}


	cout << "Done" << endl;
	_getch();
	return 0;
}




