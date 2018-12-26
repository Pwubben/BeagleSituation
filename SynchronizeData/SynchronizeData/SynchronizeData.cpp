#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include <string>
#include <sstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "trim.h"

int main()
{
	//Video trimming
	{
		//	// SS1 SC
		//	{
		//		std::stringstream ss;
		//		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		//		std::string file = "ss1 sc.mp4";
		//		ss << path << file;
		//		std::string s = ss.str();

		//		cv::VideoCapture capture(s);

		//		int unPauseFrame;
		//		unPause(capture, unPauseFrame);

		//		double sc_startTime = 31.0;
		//		trim(capture, sc_startTime, 118.0, unPauseFrame, "ss1_sc.avi");
		//	}

		//	//SS3 SC
		//	{
		//		std::stringstream ss;
		//		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		//		std::string file = "ss3 sc.mp4";
		//		ss << path << file;
		//		std::string s = ss.str();

		//		cv::VideoCapture capture(s);

		//		int unPauseFrame;
		//		unPause(capture, unPauseFrame);

		//		double sc_startTime = 31.0;
		//		trim(capture, sc_startTime, 118.0, unPauseFrame, "ss3_sc.avi");
		//	}

		//	// SS1 MC
		//	{
		//		std::stringstream ss;
		//		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		//		std::string file = "ss1 mc.mp4";
		//		ss << path << file;
		//		std::string s = ss.str();

		//		cv::VideoCapture capture(s);

		//		int unPauseFrame;
		//		unPause(capture, unPauseFrame);

		//		double sc_startTime = 31.0;
		//		trim(capture, sc_startTime, 114.0, unPauseFrame, "ss1_mc.avi");
		//	}

		//	// SS3 MC
		//	{
		//		std::stringstream ss;
		//		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		//		std::string file = "ss3 mc.mp4";
		//		ss << path << file;
		//		std::string s = ss.str();

		//		cv::VideoCapture capture(s);

		//		int unPauseFrame;
		//		unPause(capture, unPauseFrame);

		//		double sc_startTime = 49.0;
		//		trim(capture, sc_startTime, 132.0, unPauseFrame, "ss3_mc.avi");
		//	}
		//	// SS1 MC MB
		//	{
		//		std::stringstream ss;
		//		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		//		std::string file = "ss1 mc mb.mp4";
		//		ss << path << file;
		//		std::string s = ss.str();

		//		cv::VideoCapture capture(s);

		//		int unPauseFrame;
		//		unPause(capture, unPauseFrame);

		//		double sc_startTime = 0.0;
		//		trim(capture, sc_startTime, 84.0, unPauseFrame, "ss1_mc_mb.avi");
		//	}

		//	// SS3 MC MB
		//	{
		//		std::stringstream ss;
		//		std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
		//		std::string file = "ss3 mc mb 2.mp4";
		//		ss << path << file;
		//		std::string s = ss.str();

		//		cv::VideoCapture capture(s);

		//		int unPauseFrame;
		//		unPause(capture, unPauseFrame);

		//		double sc_startTime = 0.0;
		//		trim(capture, sc_startTime, 84.0, unPauseFrame, "ss3_mc_mb.avi");
		//	}
			// SS1 SC MB
			//{
			//	std::stringstream ss;
			//	std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
			//	std::string file = "ss1 sc mb.mp4";
			//	ss << path << file;
			//	std::string s = ss.str();

			//	cv::VideoCapture capture(s);

			//	int unPauseFrame;
			//	unPause(capture, unPauseFrame);

			//	double sc_startTime = 0.0;
			//	trim(capture, sc_startTime, 86.0, unPauseFrame, "ss1_sc_mb.avi");
			//}

			//// SS3 SC MB
			/*{
				std::stringstream ss;
				std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
				std::string file = "ss3 sc mb.mp4";
				ss << path << file;
				std::string s = ss.str();

				cv::VideoCapture capture(s);

				int unPauseFrame;
				unPause(capture, unPauseFrame);

				double sc_startTime = 0.0;
				trim(capture, sc_startTime, 86.0, unPauseFrame, "ss3_sc_mb.avi");
			}*/
	}

	// IMU Data processing
	// Load data file
	std::stringstream ss;
	std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
	std::string file = "ss1 mc mb.log";
	ss << path << file;
	std::string s = ss.str();
	
	std::vector<double> ROTvec;
	std::vector<double> HDTvec;

	IMUData(s, ROTvec, HDTvec);
	_getch();
    return 0;
}

