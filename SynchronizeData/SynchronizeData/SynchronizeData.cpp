#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include <string>
#include <sstream>
#include <vector>
#include <memory>
#include "opencv2/opencv.hpp"
#include "trim.h"
#include "DetectAlgorithms.h"
#include "SynchronizeData.h"

int main()
{
	std::string path = "G:\\Afstuderen\\Nautis Run 5-3-19\\";
	//Video trimming
	{
		// Load data file
		std::stringstream ss;
		std::stringstream dd;
		std::string file = "SS1_1289_Beagle.log";
		std::string des = "SS1_1289_Beagle.csv";
		ss << path << file;
		dd << path << des;
		std::string s = ss.str();
		std::string d = dd.str();

		double vidSize = IMUData(s, d);

		std::stringstream ff;
		file = "SS1_1289_Sync.mp4";
		ff << path << file;
		std::string f = ff.str();

		cv::VideoCapture capture(f);

		int unPauseFrame;
		unPause(capture, unPauseFrame);

		cv::VideoCapture capture2(f);

		double sc_startTime = 0.0;
		vidSize = 1695;
		trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 5-3-19\\SS1_1T_0503.avi");

	}

	{
		std::stringstream ss;
		std::stringstream dd;
		std::string file = "SS3_1289_Beagle.log";
		std::string des = "SS3_1289_Beagle.csv";
		ss << path << file;
		dd << path << des;
		std::string s = ss.str();
		std::string d = dd.str();

		double vidSize = IMUData(s, d);


		//std::stringstream ff;
		//file = "SS3_1289_Sync.mp4";
		//ff << path << file;
		//std::string f = ff.str();

		//cv::VideoCapture capture(f);

		//int unPauseFrame;
		//unPause(capture, unPauseFrame);

		//cv::VideoCapture capture2(f);

		//double sc_startTime = 0.0;
		//vidSize = 1695;
		//trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 5-3-19\\SS3_1T.avi");



	// IMU Data processing
	//{
	//// Load data file
	//std::stringstream ss;
	//std::stringstream dd;
	//std::string path = "F:\\Nautis Run 13-3-19\\";
	//std::string file = "SS3_1T_Target.log";
	//std::string des = "SS3_1T_Target.csv";
	//ss << path << file;
	//dd << path << des;
	//std::string s = ss.str();
	//std::string d = dd.str();

	//double vidSize = IMUData(s, d);
	//
	//// SS1 SC MB
	//
	//std::stringstream ff;
	//path = "G:\\Afstuderen\\Nautis Run 13-3\\";
	//file = "SS3_1T.mp4";
	//ff << path << file;
	//std::string f = ff.str();

	//cv::VideoCapture capture(f);

	//int unPauseFrame;
	//unPause(capture, unPauseFrame);

	//cv::VideoCapture capture2(f);

	//double sc_startTime = 0.0;
	//vidSize = 2559;
	//trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 13-3\\SS3_1T.avi");
	//
	//}

	//{
	//	std::stringstream ss;
	//	std::stringstream dd;
	//	std::string path = "F:\\Nautis Run 13-3-19\\";
	//	std::string file = "SS1_1T_Target.log";
	//	std::string des = "SS1_1T_Target.csv";
	//	ss << path << file;
	//	dd << path << des;
	//	std::string s = ss.str();
	//	std::string d = dd.str();

	//	double vidSize = IMUData(s, d);

		std::stringstream ff;
		path = "G:\\Afstuderen\\Nautis Run 5-3-19\\";
		file = "SS1_1T.mp4";
		ff << path << file;
		std::string f = ff.str();

		cv::VideoCapture capture(f);

		int unPauseFrame;
		unPause(capture, unPauseFrame);

		cv::VideoCapture capture2(f);

		double sc_startTime = 0.0;
		vidSize = 2560;
		trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 5-3-19\\SS1_1T.avi");
	}
	//
	//}

	//{
	//	std::stringstream ss;
	//	std::stringstream dd;
	//	std::string path = "F:\\Nautis Run 13-3-19\\";
	//	std::string file = "SS5_1T_Target.log";
	//	std::string des = "SS5_1T_Target.csv";
	//	ss << path << file;
	//	dd << path << des;
	//	std::string s = ss.str();
	//	std::string d = dd.str();

	//	double vidSize = IMUData(s, d);

	//	std::stringstream ff;
	//	path = "G:\\Afstuderen\\Nautis Run 13-3\\";
	//	file = "SS5_1T.mp4";
	//	ff << path << file;
	//	std::string f = ff.str();

	//	cv::VideoCapture capture(f);

	//	int unPauseFrame;
	//	unPause(capture, unPauseFrame);

	//	cv::VideoCapture capture2(f);
	//	unPauseFrame = 33;
	//	vidSize = 2564;
	//	double sc_startTime = 0.0;
	//	trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 13-3\\SS5_1T.avi");
	//}

	//{
	//	std::stringstream ss;
	//	std::stringstream dd;
	//	std::string path = "F:\\Nautis Run 13-3-19\\";
	//	std::string file = "SS3_2T_Target1.log";
	//	std::string des = "SS3_2T_Target1.csv";
	//	ss << path << file;
	//	dd << path << des;
	//	std::string s = ss.str();
	//	std::string d = dd.str();

	//	double vidSize = IMUData(s, d);

	//	std::stringstream sss;
	//	std::stringstream ddd;
	//	file = "SS3_2T_Target2.log";
	//	des = "SS3_2T_Target2.csv";
	//	sss << path << file;
	//	ddd << path << des;
	//	s = sss.str();
	//	d = ddd.str();

	//	vidSize = IMUData(s, d);

	//	std::stringstream ff;
	//	path = "G:\\Afstuderen\\Nautis Run 13-3\\";
	//	file = "SS3_2T.mp4";
	//	ff << path << file;
	//	std::string f = ff.str();

	//	cv::VideoCapture capture(f);

	//	int unPauseFrame;
	//	unPause(capture, unPauseFrame);

	//	cv::VideoCapture capture2(f);

	//	double sc_startTime = 0.0;
	//	vidSize = 2562;
	//	trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 13-3\\SS3_2T.avi");


	//}

	//{
	//	std::stringstream ss;
	//	std::stringstream dd;
	//	std::string path = "F:\\Nautis Run 13-3-19\\";
	//	std::string file = "SS1_2T_Target1.log";
	//	std::string des = "SS1_2T_Target1.csv";
	//	ss << path << file;
	//	dd << path << des;
	//	std::string s = ss.str();
	//	std::string d = dd.str();

	//	double vidSize = IMUData(s, d);

	//	std::stringstream sss;
	//	std::stringstream ddd;
	//	file = "SS1_2T_Target2.log";
	//	des = "SS1_2T_Target2.csv";
	//	sss << path << file;
	//	ddd << path << des;
	//	s = sss.str();
	//	d = ddd.str();

	//	vidSize = IMUData(s, d);

	//	std::stringstream ff;
	//	path = "G:\\Afstuderen\\Nautis Run 13-3\\";
	//	file = "SS1_2T.mp4";
	//	ff << path << file;
	//	std::string f = ff.str();

	//	cv::VideoCapture capture(f);

	//	int unPauseFrame;
	//	unPause(capture, unPauseFrame);

	//	cv::VideoCapture capture2(f);
	//	vidSize = 2562;
	//	double sc_startTime = 0.0;
	//	trim(capture2, sc_startTime, vidSize, unPauseFrame, "G:\\Afstuderen\\Nautis Run 13-3\\SS1_2T.avi");

	//}

	_getch();
    return 0;
}


