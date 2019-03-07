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
	//Video trimming
	{
		// Generate Ground truth bounding box vectors

		// SS3 SC Ground Truth
		/*{
			std::stringstream ss;
			std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
			std::string file = "GroundTruth.avi";
			ss << path << file;
			std::string s = ss.str();
			cv::VideoCapture capture(s);

			std::vector<cv::Rect> ss3_scGroundTruth;
			GroundTruth(capture, ss3_scGroundTruth);
			std::ofstream ss3_scGroundTruthFile("F:\\Afstuderen\\Afstuderen\\Videos\\ss3_scGroundTruth.csv");


			for (int n = 0; n < ss3_scGroundTruth.size(); n++)
			{
				ss3_scGroundTruthFile << ss3_scGroundTruth[n].x << "," << ss3_scGroundTruth[n].y << "," << ss3_scGroundTruth[n].width << "," << ss3_scGroundTruth[n].height << endl;
			}
			ss3_scGroundTruthFile.close();
		}*/
		//Check synchonization result
		/*{
			{
				cv::Mat src;
				std::stringstream ss;
				std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
				std::string file = "ss3_sc.avi";
				ss << path << file;
				std::string s = ss.str();
				cv::VideoCapture capture(s);
				for (int i = 0; i < 1150; i++) {
					capture >> src;
					imshow("src1", src);
				}
			}
			{
				cv::Mat src;
				std::stringstream ss;
				std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
				std::string file = "GroundTruth.avi";
				ss << path << file;
				std::string s = ss.str();
				cv::VideoCapture capture(s);
				for (int i = 0; i < 1150; i++) {
					capture >> src;
					imshow("src2", src);
				}
			}
			cv::waitKey(0);
		}*/


			// Ground Truth 
			/*{
				std::stringstream ss;
				std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
				std::string file = "ss3-mk.mp4";
				ss << path << file;
				std::string s = ss.str();

				cv::VideoCapture capture(s);

				int unPauseFrame;
				unPause(capture, unPauseFrame);

				double sc_startTime = 31.0;
				trim(capture, sc_startTime, 118.0, unPauseFrame, "F:\\Afstuderen\\Afstuderen\\Videos\\GroundTruth.avi");
			}*/
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

			//SS3 SC
		/*	{
				std::stringstream ss;
				std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
				std::string file = "ss3 sc.mp4";
				ss << path << file;
				std::string s = ss.str();

				cv::VideoCapture capture(s);

				int unPauseFrame;
				unPause(capture, unPauseFrame);

				double sc_startTime = 31.0;
				trim(capture, sc_startTime, 118.0, unPauseFrame, "ss3_sc.avi",2);
			}*/

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
			/*{
				std::stringstream ss;
				std::string path = "F:\\Afstuderen\\";
				std::string file = "ss1 sc mb.mp4";
				ss << path << file;
				std::string s = ss.str();

				cv::VideoCapture capture(s);

				int unPauseFrame;
				unPause(capture, unPauseFrame);

				double sc_startTime = 0.0;
				trim(capture, sc_startTime, 86.0, unPauseFrame, "ss1_sc_mb_temp.avi");
			}*/

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


	// C++ program to print all combinations of size 
	// k of elements in set 1..n 

	
	// IMU Data processing
	{
	// Load data file
	std::stringstream ss;
	std::stringstream dd;
	std::string path = "F:\\Nautis Run 5-3-19\\";
	std::string file = "SS1_1289_2T_Target2.log";
	std::string des = "SS1_1289_2T_Target2.csv";
	ss << path << file;
	dd << path << des;
	std::string s = ss.str();
	std::string d = dd.str();

	double vidSize = IMUData(s, d);
	
	// SS1 SC MB
	
	/*std::stringstream ff;
	path = "F:\\Nautis Run 5-3-19\\";
	file = "SS1_1289_2T_Sync.mp4";
	ff << path << file;
	std::string f = ff.str();

	cv::VideoCapture capture(f);

	int unPauseFrame;
	unPause(capture, unPauseFrame);

	cv::VideoCapture capture2(f);

	double sc_startTime = 0.0;
	trim(capture2, sc_startTime, vidSize, unPauseFrame, "F:\\Nautis Run 5-3-19\\SS1_1289_2T_Sync.avi");*/
	
	}



	_getch();
    return 0;
}


