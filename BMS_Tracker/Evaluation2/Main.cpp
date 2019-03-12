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
	/******///Deze veranderen/***********/
	/************************************/
	std::string path = "F:\\Nautis Run 5-3-19\\";
	/************************************/

	//waitKey(0);
	double duration = static_cast<double>(cv::getTickCount());

	////Data Generation
	{
		std::string File = "SS1_1289_2T_Sync.avi";
		std::string groundTruthFile = "ss3_sc_mbGroundtruth.csv";
		std::string targetFile = "SS1_1289_2T_Target1_interp.csv";
		std::string beagleFile = "SS1_1289_2T_Beagle_interp.csv";
		std::string radarFile = "SS1_1289_2T_radar_RadVel_noise.csv";

		std::string beagleDes = "SS1_1289_2T_Beagle_stateData.csv";
		std::string targetDes = "SS1_1289_2T_Target1_stateData.csv";

		std::string stateResultFile = "SS1_1289_2T_TargetStateResult_NCE42.csv";

		//Evaluation Settings
		evaluationSettings settings({
			false,		//Camera utilization
			{ 1,2,3,5 },  //Dynamic models
			2,          //Object Choice
			5.5         //Detection threshold
		});

		Detection* detection = new Detection(settings);
		detection->run(path, File, groundTruthFile, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);
	}

	//Data Generation
	{
		std::string File = "SS1_1289_2T_Sync.avi";
		std::string groundTruthFile = "ss3_sc_mbGroundtruth.csv";
		std::string targetFile = "SS1_1289_2T_Target1_interp.csv";
		std::string beagleFile = "SS1_1289_2T_Beagle_interp.csv";
		std::string radarFile = "SS1_1289_2T_radar_RadVel_noise.csv";

		std::string beagleDes = "SS1_1289_2T_Beagle_stateData.csv";
		std::string targetDes = "SS1_1289_2T_Target1_stateData.csv";

		std::string stateResultFile = "SS1_1289_2T_CE4255.csv"; //Camera EKF object 0 5.5 

		//Evaluation Settings
		evaluationSettings settings({
		true,		//Camera utilization
		{5,6,7,8},  //Dynamic models
		2,          //Object Choice
		5.5         //Detection threshold
		});

		Detection* detection = new Detection(settings);
		detection->run(path, File, groundTruthFile,beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile,2);
	}

	

	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	std::cout << "Duration: " << duration << std::endl;

	cout << "Done" << endl;
	_getch();
	return 0;
}




