#include <iostream>
#include "GnuGraph.h"
#include <stdio.h>
#include <ctime>
#include <conio.h>
#include <vector>
#include <future>
#include <omp.h>
#include "opencv2/opencv.hpp"
#include "BMS.h"
#include "Tracker.h"

using namespace cv;
using namespace std;

void RunAsync(evaluationSettings settings, std::string path, std::string File, std::string beagleFile, std::string radarFile, std::string targetFile, std::string beagleDes, std::string targetDes, std::string resultDes, int targets) {
	double fov;
	if (File == "SR_SS1.avi" || File == "SR_SS3.avi")
		fov = 85.0;
	else
		fov = 90.0;

	Detection(settings).run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, resultDes, targets,fov);
}

int main(int argc, char* argv[])
{

	std::string path = "G:\\Afstuderen\\Nautis Run 13-3\\Data\\";

	{
		std::string File = "SS1_2T.avi";

		std::string radarFile = "SS1_2T_Rad.csv";

		std::string targetFile = "SS1_2T_Target1_interp.csv";
		std::string beagleFile = "SS1_2T_Beagle_interp.csv";
		std::string beagleDes = "SS1_2T_Beagle_stateData.csv";
		std::string targetDes = "SS1_2T_Target1_stateData.csv";
		std::string video = "SS1_2T";
		//Image size
		{

			//std::string stateResultFile = Util::giveName(video, 4, 800);
			////Evaluation Settings
			//evaluationSettings settings6({
			//	true,		//Camera utilization
			//	{6,7,8 },  //Dynamic models
			//	2,          //Object Choice
			//	4,         //Detection threshold
			//	800,
			//	1
			//});//Camera, EKF5678 4.5
			//RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);

			//stateResultFile = Util::giveName(video, 0, 800);
			////Evaluation Settings
			//evaluationSettings settings7({
			//	false,		//Camera utilization
			//	{ 5,6,7,8 },  //Dynamic models
			//	2,          //Object Choice
			//	4,         //Detection threshold
			//	800,
			//	1
			//});//Camera, EKF5678 4.5
			//RunAsync(settings7, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);
		}
	}

	//path = "G:\\Afstuderen\\Nautis Run 5-3-19\\Requisites Data\\";
	{
		std::string File = "SS1_0503_2T.avi";

		std::string radarFile = "SS1_0503_2T_Rad.csv";

		std::string targetFile = "SS1_0503_2T_Target2_interp.csv";
		std::string beagleFile = "SS1_0503_2T_Beagle_interp.csv";
		std::string beagleDes = "SS1_0503_2T_Beagle_stateData.csv";
		std::string targetDes = "SS1_0503_2T_Target2_stateData.csv";
		std::string video = "SS1_0503_2T";
		//Image size
		{

			//std::string stateResultFile = Util::giveName(video, 4, 800);
			////Evaluation Settings
			//evaluationSettings settings6({
			//	true,		//Camera utilization
			//	{ 6,7,8 },  //Dynamic models
			//	2,          //Object Choice
			//	4,         //Detection threshold
			//	800,
			//	1
			//});//Camera, EKF5678 4.5
			//RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);

			//stateResultFile = Util::giveName(video, 0, 800);
			////Evaluation Settings
			//evaluationSettings settings7({
			//	false,		//Camera utilization
			//	{ 5,6,7,8 },  //Dynamic models
			//	2,          //Object Choice
			//	4,         //Detection threshold
			//	800,
			//	1
			//});//Camera, EKF5678 4.5
			//RunAsync(settings7, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);
		}
	}

	/******///Deze veranderen/***********/
	/************************************/
	 path = "G:\\Afstuderen\\Nautis Run 13-3\\Data\\";
	 //path = "G:\\Afstuderen\\";
	/************************************/

	//waitKey(0);
	double duration = static_cast<double>(cv::getTickCount());

	//Data Generation
	{
		std::string File = "SR_SS1.avi";

		std::string radarFile = "SR_SS1_Rad.csv";

		std::string targetFile = "SR_SS1_Target_interp.csv";
		std::string beagleFile = "SR_SS1_Beagle_interp.csv";
		std::string beagleDes = "SR_SS1_Beagle_stateData2.csv";
		std::string targetDes = "SR_SS1_Target_stateData2.csv";
		std::string video = "SR_SS1";


	//	//Image size
	{

			//for (double threshold = 4; threshold < 4.1; threshold += 0.5) {
			//	for (double imageSize = 800; imageSize < 801; imageSize += 400) {
			//		std::string stateResultFile = Util::giveName(video, threshold, imageSize);
			//		//Evaluation Settings
			//		evaluationSettings settings6({
			//			true,		//Camera utilization
			//			{6,7,8},  //Dynamic models
			//			2,          //Object Choice
			//			threshold,         //Detection threshold
			//			imageSize,
			//			1
			//		});//Camera, EKF5678 4.5
			//		RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
 		//		}
			//}

			//VarianceFactor
				//#pragma omp parallel for
			for (double varianceFactor = 1.0; varianceFactor <1.41; varianceFactor += 0.02) {
				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
				//Evaluation Settings
				evaluationSettings settings1({
					true,		//Camera utilization
					{ 6,10,11 },  //Dynamic models
					2,          //Object Choice
					4,         //Detection threshold
					800,//Max dimension
					varianceFactor//Variance factor
				});//Camera, EKF5

				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			}

			//}
	////		//Different models with camera
			//	{
					//std::string stateResultFile = Util::giveName(video, 80, 800);
					////Evaluation Settings
					//evaluationSettings settings1({
					//	true,		//Camera utilization
					//	{6,10,11},  //Dynamic models
					//	2,          //Object Choice
					//	4,         //Detection threshold
					//	800,//Max dimension
					//	1//Variance factor
					//});//Camera, EKF5
					//auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//		stateResultFile = Util::giveName(video, 10, 800);
			//		//Evaluation Settings
			//		evaluationSettings settings2({
			//			true,		//Camera utilization
			//			{ 0 },  //Dynamic models
			//			0,         //Object Choice
			//			3.5,         //Detection threshold
			//			800,//Max dimension
			//			1//Variance factor
			//		});//Camera, EKF5
			//		auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//		stateResultFile = Util::giveName(video, 1250, 800);
			//		//Evaluation Settings
			//		evaluationSettings settings3({
			//			true,		//Camera utilization
			//			{0,1,2,5},  //Dynamic models
			//			2,          //Object Choice
			//			3.5,         //Detection threshold
			//			800,//Max dimension
			//			1//Variance factor
			//		});//Camera, EKF5
			//		auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//std::string stateResultFile = Util::giveName(video, 6780, 800);
					////Evaluation Settings
					//evaluationSettings settings4({
					//	true,		//Camera utilization
					//	{ 6,7,8 },  //Dynamic models
					//	2,          //Object Choice
					//	4,         //Detection threshold
					//	800,//Max dimension
					//	1//Variance factor
					//});//Camera, EKF5
					//auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			//		stateResultFile = Util::giveName(video, 56780, 800);
			//		//Evaluation Settings
			//		evaluationSettings settings5({
			//			true,		//Camera utilization
			//			{5, 6,7,8 },  //Dynamic models
			//			2,          //Object Choice
			//			3.5,         //Detection threshold
			//			800,//Max dimension
			//			1//Variance factor
			//		});//Camera, EKF5
			//		auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			//	}
			//	//Different models without camera
					//{
					//	std::string stateResultFile = Util::giveName(video, 80, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings10({
					//		true,		//Camera utilization
					//		{ 6,10,11 },  //Dynamic models
					//		2,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f10 = std::async(std::launch::async, RunAsync, settings10, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	stateResultFile = Util::giveName(video, 6780, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings4({
					//		true,		//Camera utilization
					//		{ 6,7,8 },  //Dynamic models
					//		2,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
					//	
					//	stateResultFile = Util::giveName(video, 50, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings1({
					//		true,		//Camera utilization
					//		{ 5 },  //Dynamic models
					//		1,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 10, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings2({
					//		true,		//Camera utilization
					//		{ 0 },  //Dynamic models
					//		0,         //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	//stateResultFile = Util::giveName(video, 1250, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings3({
					//	//	true,		//Camera utilization
					//	//	{ 0,1,2,5 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	3.5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 6780, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings40({
					//		true,		//Camera utilization
					//		{ 6,7,8 },  //Dynamic models
					//		2,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f40 = std::async(std::launch::async, RunAsync, settings40, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	//stateResultFile = Util::giveName(video, 56780, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings5({
					//	//	true,		//Camera utilization
					//	//	{ 5, 6,7,8 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	3.5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
					//}
					////Different models without camera
					//{
					//	std::string stateResultFile = Util::giveName(video, 55, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings1({
					//		false,		//Camera utilization
					//		{ 5 },  //Dynamic models
					//		1,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 0, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings2({
					//		false,		//Camera utilization
					//		{ 0 },  //Dynamic models
					//		0,         //Object Choice
					//		5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	//stateResultFile = Util::giveName(video, 0125, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings3({
					//	//	false,		//Camera utilization
					//	//	{ 0,1,2,5 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 678, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings4({
					//		false,		//Camera utilization
					//		{ 6,7,8 },  //Dynamic models
					//		2,          //Object Choice
					//		5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	//stateResultFile = Util::giveName(video, 5678, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings5({
					//	//	false,		//Camera utilization
					//	//	{ 5, 6,7,8 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
					//}
			}
		}
		////	
		//	////Data Generation
		{
			std::string File = "SR_SS3.avi";

			std::string radarFile = "SR_SS3_Rad.csv";

			std::string targetFile = "SR_SS3_Target_interp.csv";
			std::string beagleFile = "SR_SS3_Beagle_interp.csv";
			std::string beagleDes = "SR_SS3_Beagle_stateData2.csv";
			std::string targetDes = "SR_SS3_Target_stateData2.csv";
			std::string video = "SR_SS3";
			//Image size
			{

				//for (double threshold = 4; threshold < 4.1; threshold += 0.5) {
				//	for (double imageSize = 800; imageSize < 801; imageSize += 400) {
				//		std::string stateResultFile = Util::giveName(video, threshold, imageSize);
				//		//Evaluation Settings
				//		evaluationSettings settings6({
				//			false,		//Camera utilization
				//			{ 6,7,8 },  //Dynamic models
				//			2,          //Object Choice
				//			threshold,         //Detection threshold
				//			imageSize,
				//			1
				//		});//Camera, EKF5678 4.5
				//		RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//	}
				//}
				for (double varianceFactor = 1.0; varianceFactor <1.41; varianceFactor += 0.02) {
					std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
					//Evaluation Settings
					evaluationSettings settings1({
						true,		//Camera utilization
						{ 6,10,11 },  //Dynamic models
						2,          //Object Choice
						4,         //Detection threshold
						800,//Max dimension
						varianceFactor//Variance factor
					});//Camera, EKF5

					RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}
				//}
				//}

				
				//{
				//	std::string stateResultFile = Util::giveName(video, 80, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings10({
				//		true,		//Camera utilization
				//		{ 6,10,11 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f10 = std::async(std::launch::async, RunAsync, settings10, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	stateResultFile = Util::giveName(video, 6780, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings4({
				//		true,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	stateResultFile = Util::giveName(video, 50, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings1({
				//		true,		//Camera utilization
				//		{ 5 },  //Dynamic models
				//		1,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 10, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings2({
				//		true,		//Camera utilization
				//		{ 0 },  //Dynamic models
				//		0,         //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	//stateResultFile = Util::giveName(video, 1250, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings3({
				//	//	true,		//Camera utilization
				//	//	{ 0,1,2,5 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	3.5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 6780, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings40({
				//		true,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f40 = std::async(std::launch::async, RunAsync, settings40, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	//stateResultFile = Util::giveName(video, 56780, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings5({
				//	//	true,		//Camera utilization
				//	//	{ 5, 6,7,8 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	3.5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
				////Different models without camera
				//{
				//	std::string stateResultFile = Util::giveName(video, 55, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings1({
				//		false,		//Camera utilization
				//		{ 5 },  //Dynamic models
				//		1,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 0, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings2({
				//		false,		//Camera utilization
				//		{ 0 },  //Dynamic models
				//		0,         //Object Choice
				//		5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	//stateResultFile = Util::giveName(video, 0125, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings3({
				//	//	false,		//Camera utilization
				//	//	{ 0,1,2,5 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 678, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings4({
				//		false,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	//stateResultFile = Util::giveName(video, 5678, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings5({
				//	//	false,		//Camera utilization
				//	//	{ 5, 6,7,8 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
				//{
				//			std::string stateResultFile = Util::giveName(video, 0, 800);
				//			//Evaluation Settings
				//			evaluationSettings settings6({
				//				false,		//Camera utilization
				//				{ 5,6,7,8 },  //Dynamic models
				//				2,          //Object Choice
				//				0,         //Detection threshold
				//				800,
				//				1
				//			});//Camera, EKF5678 4.5
				//			RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
			//}
		}
		//////VarianceFactor
		////#pragma omp parallel for 
		////{
		////	for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
		////		std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
		////		//Evaluation Settings
		////		evaluationSettings settings1({
		////			true,		//Camera utilization
		////			{ 5,6,7,8 },  //Dynamic models
		////			2,          //Object Choice
		////			5,         //Detection threshold
		////			800,//Max dimension
		////			varianceFactor//Variance factor
		////		});//Camera, EKF5
		////		RunAsync( settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
		////	}
		////}
		////	
		}
		////
		{
			std::string File = "SS1_1T.avi";

			std::string radarFile = "SS1_1T_Rad.csv";

			std::string targetFile = "SS1_1T_Target_interp.csv";
			std::string beagleFile = "SS1_1T_Beagle_interp.csv";
			std::string beagleDes = "SS1_1T_Beagle_stateData2.csv";
			std::string targetDes = "SS1_1T_Target_stateData2.csv";
			std::string video = "SS1_1T";

			//Image size
			{


				//for (double threshold = 4; threshold < 4.1; threshold += 0.5) {
				//	for (double imageSize = 800; imageSize < 801; imageSize += 400) {
				//		std::string stateResultFile = Util::giveName(video, threshold, imageSize);
				//		//Evaluation Settings
				//		evaluationSettings settings6({
				//			true,		//Camera utilization
				//			{ 6,7,8 },  //Dynamic models
				//			2,          //Object Choice
				//			threshold,         //Detection threshold
				//			imageSize,
				//			1
				//		});//Camera, EKF5678 4.5
				//		RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//	}
				//}

		//		//VarianceFactor

				for (double varianceFactor = 1.0; varianceFactor <1.41; varianceFactor += 0.02) {
					std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
					//Evaluation Settings
					evaluationSettings settings1({
						true,		//Camera utilization
						{ 6,10,11 },  //Dynamic models
						2,          //Object Choice
						4,         //Detection threshold
						800,//Max dimension
						varianceFactor//Variance factor
					});//Camera, EKF5

					RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}

			//Different models with camera
				//{
				//	std::string stateResultFile = Util::giveName(video, 80, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings10({
				//		true,		//Camera utilization
				//		{ 6,10,11 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f10 = std::async(std::launch::async, RunAsync, settings10, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	stateResultFile = Util::giveName(video, 6780, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings4({
				//		true,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	stateResultFile = Util::giveName(video, 50, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings1({
				//		true,		//Camera utilization
				//		{ 5 },  //Dynamic models
				//		1,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 10, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings2({
				//		true,		//Camera utilization
				//		{ 0 },  //Dynamic models
				//		0,         //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	//stateResultFile = Util::giveName(video, 1250, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings3({
				//	//	true,		//Camera utilization
				//	//	{ 0,1,2,5 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	3.5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 6780, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings40({
				//		true,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f40 = std::async(std::launch::async, RunAsync, settings40, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	//stateResultFile = Util::giveName(video, 56780, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings5({
				//	//	true,		//Camera utilization
				//	//	{ 5, 6,7,8 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	3.5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
				////Different models without camera
				//{
				//	std::string stateResultFile = Util::giveName(video, 55, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings1({
				//		false,		//Camera utilization
				//		{ 5 },  //Dynamic models
				//		1,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 0, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings2({
				//		false,		//Camera utilization
				//		{ 0 },  //Dynamic models
				//		0,         //Object Choice
				//		5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	//stateResultFile = Util::giveName(video, 0125, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings3({
				//	//	false,		//Camera utilization
				//	//	{ 0,1,2,5 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 678, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings4({
				//		false,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	//stateResultFile = Util::giveName(video, 5678, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings5({
				//	//	false,		//Camera utilization
				//	//	{ 5, 6,7,8 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
			}
		}

		//Data Generation
		{
			std::string File = "SS3_1T.avi";

			std::string radarFile = "SS3_1T_Rad.csv";

			std::string targetFile = "SS3_1T_Target_interp.csv";
			std::string beagleFile = "SS3_1T_Beagle_interp.csv";
			std::string beagleDes = "SS3_1T_Beagle_stateData2.csv";
			std::string targetDes = "SS3_1T_Target_stateData2.csv";
			std::string video = "SS3_1T";

			//Image size
			{
				
				//for (double threshold = 4; threshold < 4.1; threshold += 0.5) {
				//	for (double imageSize = 800; imageSize < 801; imageSize += 400) {
				//		std::string stateResultFile = Util::giveName(video, threshold, imageSize);
				//		//Evaluation Settings
				//		evaluationSettings settings6({
				//			false,		//Camera utilization
				//			{ 6,7,8 },  //Dynamic models
				//			2,          //Object Choice
				//			threshold,         //Detection threshold
				//			imageSize,
				//			1
				//		});//Camera, EKF5678 4.5
				//		RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//	}
				//}
			}
			for (double varianceFactor = 1.0; varianceFactor <1.41; varianceFactor += 0.02) {
				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
				//Evaluation Settings
				evaluationSettings settings1({
					true,		//Camera utilization
					{ 6,10,11 },  //Dynamic models
					2,          //Object Choice
					4,         //Detection threshold
					800,//Max dimension
					varianceFactor//Variance factor
				});//Camera, EKF5

				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			}
			{

				//{
				//	std::string stateResultFile = Util::giveName(video, 80, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings10({
				//		true,		//Camera utilization
				//		{ 6,10,11 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f10 = std::async(std::launch::async, RunAsync, settings10, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	stateResultFile = Util::giveName(video, 6780, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings4({
				//		true,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	stateResultFile = Util::giveName(video, 50, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings1({
				//		true,		//Camera utilization
				//		{ 5 },  //Dynamic models
				//		1,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 10, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings2({
				//		true,		//Camera utilization
				//		{ 0 },  //Dynamic models
				//		0,         //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	//stateResultFile = Util::giveName(video, 1250, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings3({
				//	//	true,		//Camera utilization
				//	//	{ 0,1,2,5 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	3.5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 6780, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings40({
				//		true,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f40 = std::async(std::launch::async, RunAsync, settings40, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	//stateResultFile = Util::giveName(video, 56780, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings5({
				//	//	true,		//Camera utilization
				//	//	{ 5, 6,7,8 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	3.5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
				////Different models without camera
				//{
				//	std::string stateResultFile = Util::giveName(video, 55, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings1({
				//		false,		//Camera utilization
				//		{ 5 },  //Dynamic models
				//		1,          //Object Choice
				//		3.5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 0, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings2({
				//		false,		//Camera utilization
				//		{ 0 },  //Dynamic models
				//		0,         //Object Choice
				//		5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	//stateResultFile = Util::giveName(video, 0125, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings3({
				//	//	false,		//Camera utilization
				//	//	{ 0,1,2,5 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


				//	stateResultFile = Util::giveName(video, 678, 800);
				//	//Evaluation Settings
				//	evaluationSettings settings4({
				//		false,		//Camera utilization
				//		{ 6,7,8 },  //Dynamic models
				//		2,          //Object Choice
				//		5,         //Detection threshold
				//		800,//Max dimension
				//		1//Variance factor
				//	});//Camera, EKF5
				//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

				//	//stateResultFile = Util::giveName(video, 5678, 800);
				//	////Evaluation Settings
				//	//evaluationSettings settings5({
				//	//	false,		//Camera utilization
				//	//	{ 5, 6,7,8 },  //Dynamic models
				//	//	2,          //Object Choice
				//	//	5,         //Detection threshold
				//	//	800,//Max dimension
				//	//	1//Variance factor
				//	//});//Camera, EKF5
				//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//}
			//VarianceFactor
	//#pragma omp parallel for 
	//		{
	//			for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
	//				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
	//				//Evaluation Settings
	//				evaluationSettings settings1({
	//					true,		//Camera utilization
	//					{ 5,6,7,8 },  //Dynamic models
	//					2,          //Object Choice
	//					5,         //Detection threshold
	//					800,//Max dimension
	//					varianceFactor//Variance factor
	//				});//Camera, EKF5
	//				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	//			}
			}

		}

		

		//path = "G:\\Afstuderen\\Nautis Run 5-3-19\\";

		{
			std::string File = "SS1_0503.avi";

			std::string radarFile = "SS1_0503_Rad.csv";

			std::string targetFile = "SS1_0503_Target_interp.csv";
			std::string beagleFile = "SS1_0503_Beagle_interp.csv";
			std::string beagleDes = "SS1_0503_Beagle_stateData2.csv";
			std::string targetDes = "SS1_0503_Target_stateData2.csv";
			std::string video = "SS1_0503";
			//Image size
			{
				//for (double threshold = 4; threshold < 4.1; threshold += 0.5) {
				//	for (double imageSize = 800; imageSize < 801; imageSize += 400) {
				//		std::string stateResultFile = Util::giveName(video, threshold, imageSize);
				//		//Evaluation Settings
				//		evaluationSettings settings6({
				//			false,		//Camera utilization
				//			{ 6,7,8 },  //Dynamic models
				//			2,          //Object Choice
				//			threshold,         //Detection threshold
				//			imageSize,
				//			1
				//		});//Camera, EKF5678 4.5
				//		RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//	}
				//}
				for (double varianceFactor = 1.0; varianceFactor <1.41; varianceFactor += 0.02) {
					std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
					//Evaluation Settings
					evaluationSettings settings1({
						true,		//Camera utilization
						{ 6,10,11 },  //Dynamic models
						2,          //Object Choice
						4,         //Detection threshold
						800,//Max dimension
						varianceFactor//Variance factor
					});//Camera, EKF5

					RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}
				//VarianceFactor
	//#pragma omp parallel for
	//			for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
	//				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
	//				//Evaluation Settings
	//				evaluationSettings settings1({
	//					true,		//Camera utilization
	//					{ 5,6,7,8 },  //Dynamic models
	//					2,          //Object Choice
	//					5,         //Detection threshold
	//					800,//Max dimension
	//					varianceFactor//Variance factor
	//				});//Camera, EKF5
	//
	//				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	//			}
			}
			//Different models with camera
			//{
			//	std::string stateResultFile = Util::giveName(video, 80, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings10({
			//		true,		//Camera utilization
			//		{ 6,10,11 },  //Dynamic models
			//		2,          //Object Choice
			//		3.5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f10 = std::async(std::launch::async, RunAsync, settings10, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			//	stateResultFile = Util::giveName(video, 6780, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings4({
			//		true,		//Camera utilization
			//		{ 6,7,8 },  //Dynamic models
			//		2,          //Object Choice
			//		3.5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			//	stateResultFile = Util::giveName(video, 50, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings1({
			//		true,		//Camera utilization
			//		{ 5 },  //Dynamic models
			//		1,          //Object Choice
			//		3.5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//	stateResultFile = Util::giveName(video, 10, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings2({
			//		true,		//Camera utilization
			//		{ 0 },  //Dynamic models
			//		0,         //Object Choice
			//		3.5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//	//stateResultFile = Util::giveName(video, 1250, 800);
			//	////Evaluation Settings
			//	//evaluationSettings settings3({
			//	//	true,		//Camera utilization
			//	//	{ 0,1,2,5 },  //Dynamic models
			//	//	2,          //Object Choice
			//	//	3.5,         //Detection threshold
			//	//	800,//Max dimension
			//	//	1//Variance factor
			//	//});//Camera, EKF5
			//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//	stateResultFile = Util::giveName(video, 6780, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings40({
			//		true,		//Camera utilization
			//		{ 6,7,8 },  //Dynamic models
			//		2,          //Object Choice
			//		3.5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f40 = std::async(std::launch::async, RunAsync, settings40, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			//	//stateResultFile = Util::giveName(video, 56780, 800);
			//	////Evaluation Settings
			//	//evaluationSettings settings5({
			//	//	true,		//Camera utilization
			//	//	{ 5, 6,7,8 },  //Dynamic models
			//	//	2,          //Object Choice
			//	//	3.5,         //Detection threshold
			//	//	800,//Max dimension
			//	//	1//Variance factor
			//	//});//Camera, EKF5
			//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			//}
			////Different models without camera
			//{
			//	std::string stateResultFile = Util::giveName(video, 55, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings1({
			//		false,		//Camera utilization
			//		{ 5 },  //Dynamic models
			//		1,          //Object Choice
			//		3.5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//	stateResultFile = Util::giveName(video, 0, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings2({
			//		false,		//Camera utilization
			//		{ 0 },  //Dynamic models
			//		0,         //Object Choice
			//		5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//	//stateResultFile = Util::giveName(video, 0125, 800);
			//	////Evaluation Settings
			//	//evaluationSettings settings3({
			//	//	false,		//Camera utilization
			//	//	{ 0,1,2,5 },  //Dynamic models
			//	//	2,          //Object Choice
			//	//	5,         //Detection threshold
			//	//	800,//Max dimension
			//	//	1//Variance factor
			//	//});//Camera, EKF5
			//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			//	stateResultFile = Util::giveName(video, 678, 800);
			//	//Evaluation Settings
			//	evaluationSettings settings4({
			//		false,		//Camera utilization
			//		{ 6,7,8 },  //Dynamic models
			//		2,          //Object Choice
			//		5,         //Detection threshold
			//		800,//Max dimension
			//		1//Variance factor
			//	});//Camera, EKF5
			//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			//	//stateResultFile = Util::giveName(video, 5678, 800);
			//	////Evaluation Settings
			//	//evaluationSettings settings5({
			//	//	false,		//Camera utilization
			//	//	{ 5, 6,7,8 },  //Dynamic models
			//	//	2,          //Object Choice
			//	//	5,         //Detection threshold
			//	//	800,//Max dimension
			//	//	1//Variance factor
			//	//});//Camera, EKF5
			//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			//}
		}
	
		{
			std::string File = "SS3_0503.avi";

			std::string radarFile = "SS3_0503_Rad.csv";

			std::string targetFile = "SS3_0503_Target_interp.csv";
			std::string beagleFile = "SS3_0503_Beagle_interp.csv";
			std::string beagleDes = "SS3_0503_Beagle_stateData2.csv";
			std::string targetDes = "SS3_0503_Target_stateData2.csv";
			std::string video = "SS3_0503";
			//Image size
			{
				//for (double threshold = 4; threshold < 4.1; threshold += 0.5) {
				//	for (double imageSize = 800; imageSize < 801; imageSize += 400) {
				//		std::string stateResultFile = Util::giveName(video, threshold, imageSize);
				//		//Evaluation Settings
				//		evaluationSettings settings6({
				//			true,		//Camera utilization
				//			{ 6,7,8 },  //Dynamic models
				//			2,          //Object Choice
				//			threshold,         //Detection threshold
				//			imageSize,
				//			1
				//		});//Camera, EKF5678 4.5
				//		RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				//	}
				//}
				for (double varianceFactor = 1.0; varianceFactor <1.41; varianceFactor += 0.02) {
					std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
					//Evaluation Settings
					evaluationSettings settings1({
						true,		//Camera utilization
						{ 6,10,11 },  //Dynamic models
						2,          //Object Choice
						4,         //Detection threshold
						800,//Max dimension
						varianceFactor//Variance factor
					});//Camera, EKF5

					RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}
				{
					//{
					//	std::string stateResultFile = Util::giveName(video, 80, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings10({
					//		true,		//Camera utilization
					//		{ 6,10,11 },  //Dynamic models
					//		2,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f10 = std::async(std::launch::async, RunAsync, settings10, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	stateResultFile = Util::giveName(video, 6780, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings4({
					//		true,		//Camera utilization
					//		{ 6,7,8 },  //Dynamic models
					//		2,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	stateResultFile = Util::giveName(video, 50, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings1({
					//		true,		//Camera utilization
					//		{ 5 },  //Dynamic models
					//		1,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 10, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings2({
					//		true,		//Camera utilization
					//		{ 0 },  //Dynamic models
					//		0,         //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	//stateResultFile = Util::giveName(video, 1250, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings3({
					//	//	true,		//Camera utilization
					//	//	{ 0,1,2,5 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	3.5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 6780, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings40({
					//		true,		//Camera utilization
					//		{ 6,7,8 },  //Dynamic models
					//		2,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f40 = std::async(std::launch::async, RunAsync, settings40, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	//stateResultFile = Util::giveName(video, 56780, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings5({
					//	//	true,		//Camera utilization
					//	//	{ 5, 6,7,8 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	3.5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
					//}
					////Different models without camera
					//{
					//	std::string stateResultFile = Util::giveName(video, 55, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings1({
					//		false,		//Camera utilization
					//		{ 5 },  //Dynamic models
					//		1,          //Object Choice
					//		3.5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 0, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings2({
					//		false,		//Camera utilization
					//		{ 0 },  //Dynamic models
					//		0,         //Object Choice
					//		5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	//stateResultFile = Util::giveName(video, 0125, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings3({
					//	//	false,		//Camera utilization
					//	//	{ 0,1,2,5 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


					//	stateResultFile = Util::giveName(video, 678, 800);
					//	//Evaluation Settings
					//	evaluationSettings settings4({
					//		false,		//Camera utilization
					//		{ 6,7,8 },  //Dynamic models
					//		2,          //Object Choice
					//		5,         //Detection threshold
					//		800,//Max dimension
					//		1//Variance factor
					//	});//Camera, EKF5
					//	auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

					//	//stateResultFile = Util::giveName(video, 5678, 800);
					//	////Evaluation Settings
					//	//evaluationSettings settings5({
					//	//	false,		//Camera utilization
					//	//	{ 5, 6,7,8 },  //Dynamic models
					//	//	2,          //Object Choice
					//	//	5,         //Detection threshold
					//	//	800,//Max dimension
					//	//	1//Variance factor
					//	//});//Camera, EKF5
					//	//auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
					//}
				}
			}
		}
		//VarianceFactor
//#pragma omp parallel for
//		for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
//			std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
//			//Evaluation Settings
//			evaluationSettings settings1({
//				true,		//Camera utilization
//				{ 5,6,7,8 },  //Dynamic models
//				2,          //Object Choice
//				5,         //Detection threshold
//				800,//Max dimension
//				varianceFactor//Variance factor
//			});//Camera, EKF5
//
//			RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
//		}
	//}

//}


	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	std::cout << "Duration: " << duration << std::endl;

	std::cout << "Done" << endl;
	_getch();
	return 0;
}



