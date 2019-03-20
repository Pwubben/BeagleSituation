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
	Detection* detection_ = new Detection(settings);
	detection_->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, resultDes, targets);
	delete detection_;
}

int main(int argc, char* argv[])
{
	/******///Deze veranderen/***********/
	/************************************/
	std::string path = "G:\\Afstuderen\\Nautis Run 13-3\\";
	/************************************/

	//waitKey(0);
	double duration = static_cast<double>(cv::getTickCount());

	//Data Generation
	{
		std::string File = "SR_SS1.avi";

		std::string radarFile = "SR_SS1_Rad.csv";

		std::string targetFile = "SR_SS1_Target_interp.csv";
		std::string beagleFile = "SR_SS1_Beagle_interp.csv";
		std::string beagleDes = "SR_SS1_Beagle_stateData.csv";
		std::string targetDes = "SR_SS1_Target_stateData.csv";
		std::string video = "SR_SS1";


		//Image size
		{
			#pragma omp parallel for collapse (2)
			for (double threshold = 2.5; threshold < 8.1; threshold += 0.5) {
				for (double imageSize = 800; imageSize < 1601; imageSize += 200) {
					std::string stateResultFile = Util::giveName(video, threshold, imageSize);
					//Evaluation Settings
					evaluationSettings settings6({
						true,		//Camera utilization
						{ 5,6,7,8 },  //Dynamic models
						2,          //Object Choice
						threshold,         //Detection threshold
						imageSize,
						1
					});//Camera, EKF5678 4.5
					RunAsync( settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}
			}

		//VarianceFactor
			#pragma omp parallel for
			for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
				//Evaluation Settings
				evaluationSettings settings1({
					true,		//Camera utilization
					{ 5,6,7,8 },  //Dynamic models
					2,          //Object Choice
					5,         //Detection threshold
					800,//Max dimension
					varianceFactor//Variance factor
				});//Camera, EKF5

				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			}
		}
		//Different models with camera
		{
			std::string stateResultFile = Util::giveName(video, 55, 800);
			//Evaluation Settings
			evaluationSettings settings1({
				true,		//Camera utilization
				{ 5},  //Dynamic models
				1,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			stateResultFile = Util::giveName(video, 10, 800);
			//Evaluation Settings
			evaluationSettings settings2({
				true,		//Camera utilization
				{ 0 },  //Dynamic models
				0,         //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			stateResultFile = Util::giveName(video, 1250, 800);
			//Evaluation Settings
			evaluationSettings settings3({
				true,		//Camera utilization
				{0,1,2,5},  //Dynamic models
				2,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			stateResultFile = Util::giveName(video, 6780, 800);
			//Evaluation Settings
			evaluationSettings settings4({
				true,		//Camera utilization
				{ 6,7,8 },  //Dynamic models
				2,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			stateResultFile = Util::giveName(video, 56780, 800);
			//Evaluation Settings
			evaluationSettings settings5({
				true,		//Camera utilization
				{5, 6,7,8 },  //Dynamic models
				2,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
		}
		//Different models without camera
		{
			std::string stateResultFile = Util::giveName(video, 5, 800);
			//Evaluation Settings
			evaluationSettings settings1({
				false,		//Camera utilization
				{ 5 },  //Dynamic models
				1,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			stateResultFile = Util::giveName(video, 0, 800);
			//Evaluation Settings
			evaluationSettings settings2({
				false,		//Camera utilization
				{ 0 },  //Dynamic models
				0,         //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			stateResultFile = Util::giveName(video, 0125, 800);
			//Evaluation Settings
			evaluationSettings settings3({
				false,		//Camera utilization
				{ 0,1,2,5 },  //Dynamic models
				2,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


			stateResultFile = Util::giveName(video, 678, 800);
			//Evaluation Settings
			evaluationSettings settings4({
				false,		//Camera utilization
				{ 6,7,8 },  //Dynamic models
				2,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

			stateResultFile = Util::giveName(video, 5678, 800);
			//Evaluation Settings
			evaluationSettings settings5({
				false,		//Camera utilization
				{ 5, 6,7,8 },  //Dynamic models
				2,          //Object Choice
				5,         //Detection threshold
				800,//Max dimension
				1//Variance factor
			});//Camera, EKF5
			auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
		}
	}
	
	////Data Generation
	{
		std::string File = "SR_SS3.avi";

		std::string radarFile = "SR_SS3_Rad.csv";

		std::string targetFile = "SR_SS3_Target_interp.csv";
		std::string beagleFile = "SR_SS3_Beagle_interp.csv";
		std::string beagleDes = "SR_SS3_Beagle_stateData.csv";
		std::string targetDes = "SR_SS3_Target_stateData.csv";
		std::string video = "SR_SS3";	
//Image size
{
	#pragma omp parallel for collapse (2)
	for (double threshold = 2.5; threshold < 8.1; threshold += 0.5) {
		for (double imageSize = 800; imageSize < 1601; imageSize += 200) {
			std::string stateResultFile = Util::giveName(video, threshold, imageSize);
			//Evaluation Settings
			evaluationSettings settings6({
				true,		//Camera utilization
				{ 5,6,7,8 },  //Dynamic models
				2,          //Object Choice
				threshold,         //Detection threshold
				imageSize,
				0
			});//Camera, EKF5678 4.5
			 RunAsync( settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
		}
	}
}
//VarianceFactor
#pragma omp parallel for 
{
	for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
		std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
		//Evaluation Settings
		evaluationSettings settings1({
			true,		//Camera utilization
			{ 5,6,7,8 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			varianceFactor//Variance factor
		});//Camera, EKF5
		RunAsync( settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	}
}
	
}

{
	std::string File = "SS1_1T.avi";

	std::string radarFile = "SS1_1T_Rad.csv";

	std::string targetFile = "SS1_1T_Target_interp.csv";
	std::string beagleFile = "SS1_1T_Beagle_interp.csv";
	std::string beagleDes = "SS1_1T_Beagle_stateData.csv";
	std::string targetDes = "SS1_1T_Target_stateData.csv";
	std::string video = "SS1_1T";

	//Image size
	{

#pragma omp parallel for collapse (2)
			for (double threshold = 2.5; threshold < 8.1; threshold += 0.5) {
				for (double imageSize = 800; imageSize < 1601; imageSize += 200) {
					std::string stateResultFile = Util::giveName(video, threshold, imageSize);
					//Evaluation Settings
					evaluationSettings settings6({
						true,		//Camera utilization
						{ 5,6,7,8 },  //Dynamic models
						2,          //Object Choice
						threshold,         //Detection threshold
						imageSize,
						0
					});//Camera, EKF5678 4.5
					RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}
			}


			//VarianceFactor

#pragma omp parallel for
			for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
				//Evaluation Settings
				evaluationSettings settings1({
					true,		//Camera utilization
					{ 5,6,7,8 },  //Dynamic models
					2,          //Object Choice
					5,         //Detection threshold
					800,//Max dimension
					varianceFactor//Variance factor
				});//Camera, EKF5

				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			}
		}

	//Different models with camera
	{
		std::string stateResultFile = Util::giveName(video, 55, 800);
		//Evaluation Settings
		evaluationSettings settings1({
			true,		//Camera utilization
			{ 5 },  //Dynamic models
			1,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


		stateResultFile = Util::giveName(video, 10, 800);
		//Evaluation Settings
		evaluationSettings settings2({
			true,		//Camera utilization
			{ 0 },  //Dynamic models
			0,         //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


		stateResultFile = Util::giveName(video, 1250, 800);
		//Evaluation Settings
		evaluationSettings settings3({
			true,		//Camera utilization
			{ 0,1,2,5 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


		stateResultFile = Util::giveName(video, 6780, 800);
		//Evaluation Settings
		evaluationSettings settings4({
			true,		//Camera utilization
			{ 6,7,8 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

		stateResultFile = Util::giveName(video, 56780, 800);
		//Evaluation Settings
		evaluationSettings settings5({
			true,		//Camera utilization
			{ 5, 6,7,8 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	}
	//Different models without camera
	{
		std::string stateResultFile = Util::giveName(video, 5, 800);
		//Evaluation Settings
		evaluationSettings settings1({
			false,		//Camera utilization
			{ 5 },  //Dynamic models
			1,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f1 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


		stateResultFile = Util::giveName(video, 0, 800);
		//Evaluation Settings
		evaluationSettings settings2({
			false,		//Camera utilization
			{ 0 },  //Dynamic models
			0,         //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f2 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


		stateResultFile = Util::giveName(video, 0125, 800);
		//Evaluation Settings
		evaluationSettings settings3({
			false,		//Camera utilization
			{ 0,1,2,5 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f3 = std::async(std::launch::async, RunAsync, settings3, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);


		stateResultFile = Util::giveName(video, 678, 800);
		//Evaluation Settings
		evaluationSettings settings4({
			false,		//Camera utilization
			{ 6,7,8 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f4 = std::async(std::launch::async, RunAsync, settings4, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

		stateResultFile = Util::giveName(video, 5678, 800);
		//Evaluation Settings
		evaluationSettings settings5({
			false,		//Camera utilization
			{ 5, 6,7,8 },  //Dynamic models
			2,          //Object Choice
			5,         //Detection threshold
			800,//Max dimension
			1//Variance factor
		});//Camera, EKF5
		auto f5 = std::async(std::launch::async, RunAsync, settings5, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	}

	}

	////Data Generation
	{
		std::string File = "SS3_1T.avi";

		std::string radarFile = "SS3_1T_Rad.csv";

		std::string targetFile = "SS3_1T_Target_interp.csv";
		std::string beagleFile = "SS3_1T_Beagle_interp.csv";
		std::string beagleDes = "SS3_1T_Beagle_stateData.csv";
		std::string targetDes = "SS3_1T_Target_stateData.csv";
		std::string video = "SS3_1T";

		//Image size
		{
#pragma omp parallel for collapse (2)
			for (double threshold = 2.5; threshold < 8.1; threshold += 0.5) {
				for (double imageSize = 800; imageSize < 1601; imageSize += 200) {
					std::string stateResultFile = Util::giveName(video, threshold, imageSize);
					//Evaluation Settings
					evaluationSettings settings6({
						true,		//Camera utilization
						{ 5,6,7,8 },  //Dynamic models
						2,          //Object Choice
						threshold,         //Detection threshold
						imageSize,
						1
					});//Camera, EKF5678 4.5
					RunAsync(settings6, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
				}
			}
		}
		//VarianceFactor
#pragma omp parallel for 
		{
			for (double varianceFactor = 1; varianceFactor < 1.41; varianceFactor += 0.02) {
				std::string stateResultFile = Util::giveName(video, varianceFactor, 800);
				//Evaluation Settings
				evaluationSettings settings1({
					true,		//Camera utilization
					{ 5,6,7,8 },  //Dynamic models
					2,          //Object Choice
					5,         //Detection threshold
					800,//Max dimension
					varianceFactor//Variance factor
				});//Camera, EKF5
				RunAsync(settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
			}
		}
		
	}


//	{
//		std::string File = "SS1_1T_0503.avi";
//
//		std::string radarFile = "SS1_1T_Rad.csv";
//
//		std::string targetFile = "SS1_1289_Target_interp.csv";
//		std::string beagleFile = "SS1_1289_Beagle_interp.csv";
//		std::string beagleDes = "SS1_1289_Beagle_stateData.csv";
//		std::string targetDes = "SS1_1289_Target_stateData.csv";
//		std::string video = "SS1_1T";
//
//	
//		std::string stateResultFile = Util::giveName(video, 00, 800);
//		//Evaluation Settings
//		evaluationSettings settings1({
//			false,		//Camera utilization
//			{ 5,6,7,8 },  //Dynamic models
//			2,          //Object Choice
//			5.5,         //Detection threshold
//			800,//Max dimension
//			0//Variance factor
//		});//Camera, EKF5
//		auto f6 = std::async(std::launch::async, RunAsync, settings1, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
//	
//
//		File = "SS3_1T_0503.avi";
//
//		targetFile = "SS3_1289_Target_interp.csv";
//		beagleFile = "SS3_1289_Beagle_interp.csv";
//		beagleDes = "SS3_1289_Beagle_stateData.csv";
//		targetDes = "SS3_1289_Target_stateData.csv";
//		video = "SS3_1T";
//
//
//		stateResultFile = Util::giveName(video, 00, 800);
//		//Evaluation Settings
//		evaluationSettings settings2({
//			false,		//Camera utilization
//			{ 5,6,7,8 },  //Dynamic models
//			2,          //Object Choice
//			5.5,         //Detection threshold
//			800,//Max dimension
//			0//Variance factor
//		});//Camera, EKF5
//		auto f5 = std::async(std::launch::async, RunAsync, settings2, path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
//
//
//}

	//Data Generation
	//SS5
	/*{
		std::string File = "SS5_1T.avi";

		std::string radarFile = "SS5_1T_Rad.csv";

		std::string targetFile = "SS5_1T_Target_interp.csv";
		std::string beagleFile = "SS5_1T_Beagle_interp.csv";
		std::string beagleDes = "SS5_1T_Beagle_stateData.csv";
		std::string targetDes = "SS5_1T_Target_stateData.csv";

		std::string stateResultFile = "SS1_1289_2T_TargetStateResult_NCE42.csv";
		//{
		//	//Evaluation Settings
		//	evaluationSettings settings({
		//		false,		//Camera utilization
		//		{ 5,6,7,8 },  //Dynamic models
		//		2,          //Object Choice
		//		5.5         //Detection threshold
		//	});

		//	Detection* detection = new Detection(settings);
		//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
		//}
		{
			//Evaluation Settings
			evaluationSettings settings({
				true,		//Camera utilization
				{ 5,6,7,8 },  //Dynamic models
				2,          //Object Choice
				5.5         //Detection threshold
			});

			Detection* detection = new Detection(settings);
			detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);

		}
	}*/

	////Data Generation
	//{
	//	std::string File = "SS1_2T.avi";

	//	std::string radarFile = "SS1_1289_2T_radar_RadVel_noise.csv";

	//	std::string targetFile = "SS1_2T_Target2_interp.csv";
	//	std::string beagleFile = "SS1_2T_Beagle_interp.csv";
	//	std::string beagleDes = "SS1_2T_Beagle_stateData.csv";
	//	std::string targetDes = "SS1_2T_Target2_stateData.csv";

	//	std::string stateResultFile = "SS1_1289_2T_TargetStateResult_NCE42.csv";

	//	//Evaluation Settings
	//	evaluationSettings settings({
	//		false,		//Camera utilization
	//		{ 5,6,7,8 },  //Dynamic models
	//		2,          //Object Choice
	//		5.5         //Detection threshold
	//	});

	//	Detection* detection = new Detection(settings);
	//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);
	//}

	//{
	//	std::string File = "SS3_2T.avi";

	//	std::string radarFile = "SS1_1289_2T_radar_RadVel_noise.csv";

	//	std::string targetFile = "SS3_2T_Target2_interp.csv";
	//	std::string beagleFile = "SS3_2T_Beagle_interp.csv";
	//	std::string beagleDes = "SS3_2T_Beagle_stateData.csv";
	//	std::string targetDes = "SS3_2T_Target2_stateData.csv";

	//	std::string stateResultFile = "SS1_1289_2T_TargetStateResult_NCE42.csv";

	//	//Evaluation Settings
	//	evaluationSettings settings({
	//		false,		//Camera utilization
	//		{ 5,6,7,8 },  //Dynamic models
	//		2,          //Object Choice
	//		5.5         //Detection threshold
	//	});

	//	Detection* detection = new Detection(settings);
	//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 2);
	//}

	//{
	//	std::string File = "SR_SS1.avi";

	//	std::string radarFile = "SR_SS1_Rad.csv";

	//	std::string targetFile = "SR_SS1_Target_interp.csv";
	//	std::string beagleFile = "SR_SS1_Beagle_interp.csv";
	//	std::string beagleDes = "SR_SS1_Beagle_stateData.csv";
	//	std::string targetDes = "SR_SS1_Target_stateData.csv";

	//	std::string stateResultFile = "SR_SS1_Result_NC_EKF255.csv";

	//	//Evaluation Settings
	//	evaluationSettings settings({
	//		false,		//Camera utilization
	//		{ 5,6,7,8 },  //Dynamic models
	//		2,          //Object Choice
	//		5.5         //Detection threshold
	//	});

	//	Detection* detection = new Detection(settings);
	//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	//}

	//{
	//	std::string File = "SR_SS1.avi";

	//	std::string radarFile = "SR_SS1_Rad.csv";

	//	std::string targetFile = "SR_SS1_Target_interp.csv";
	//	std::string beagleFile = "SR_SS1_Beagle_interp.csv";
	//	std::string beagleDes = "SR_SS1_Beagle_stateData.csv";
	//	std::string targetDes = "SR_SS1_Target_stateData.csv";

	//	std::string stateResultFile = "SR_SS1_Result_C_EKF255.csv";

	//	//Evaluation Settings
	//	evaluationSettings settings({
	//		true,		//Camera utilization
	//		{ 5,6,7,8 },  //Dynamic models
	//		2,          //Object Choice
	//		5.5         //Detection threshold
	//	});

	//	Detection* detection = new Detection(settings);
	//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	//}

	//{
	//	std::string File = "SR_SS3.avi";

	//	std::string radarFile = "SR_SS3_Rad.csv";

	//	std::string targetFile = "SR_SS3_Target_interp.csv";
	//	std::string beagleFile = "SR_SS3_Beagle_interp.csv";
	//	std::string beagleDes = "SR_SS3_Beagle_stateData.csv";
	//	std::string targetDes = "SR_SS3_Target_stateData.csv";

	//	std::string stateResultFile = "SR_SS3_Result_NC_EKF255.csv";

	//	//Evaluation Settings
	//	evaluationSettings settings({
	//		false,		//Camera utilization
	//		{ 5,6,7,8 },  //Dynamic models
	//		2,          //Object Choice
	//		5.5         //Detection threshold
	//	});

	//	Detection* detection = new Detection(settings);
	//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile,1);
	//}

	//{
	//	std::string File = "SR_SS3.avi";

	//	std::string radarFile = "SR_SS3_Rad.csv";

	//	std::string targetFile = "SR_SS3_Target_interp.csv";
	//	std::string beagleFile = "SR_SS3_Beagle_interp.csv";
	//	std::string beagleDes = "SR_SS3_Beagle_stateData.csv";
	//	std::string targetDes = "SR_SS3_Target_stateData.csv";

	//	std::string stateResultFile = "SR_SS3_Result_C_EKF255.csv";

	//	//Evaluation Settings
	//	evaluationSettings settings({
	//		true,		//Camera utilization
	//		{ 5,6,7,8 },  //Dynamic models
	//		2,          //Object Choice
	//		5.5         //Detection threshold
	//	});

	//	Detection* detection = new Detection(settings);
	//	detection->run(path, File, beagleFile, radarFile, targetFile, beagleDes, targetDes, stateResultFile, 1);
	//}


	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();
	std::cout << "Duration: " << duration << std::endl;

	std::cout << "Done" << endl;
	_getch();
	return 0;
}



