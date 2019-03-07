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
		std::string File = "SS3_1389_1T2.avi";
		std::string groundTruthFile = "ss3_sc_mbGroundtruth.csv";
		std::string targetFile = "SS3_1289_Target_interp.csv";
		std::string beagleFile = "SS3_1289_Beagle_interp.csv";
		std::string radarFile = "SS1_1289_Target_radar_RadVel_noise.csv";

		std::string beagleDes = "SS3_1289_1T_Beagle_stateData.csv";
		std::string targetDes = "SS3_1289_1T_Target_stateData.csv";

		Detection* detection = new Detection();
		detection->run(File, groundTruthFile,beagleFile, radarFile, targetFile, beagleDes, targetDes);
	}


	cout << "Done" << endl;
	_getch();
	return 0;
}




