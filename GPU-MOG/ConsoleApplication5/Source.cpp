#include "pch.h"
#include <opencv2/features2d/features2d.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>
#include <windows.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>
//#include "read_directory.h"


#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include <iostream>
#include <conio.h>
#include <opencv2\features2d\features2d.hpp>
#include <ctime>
#include "LBP.hpp"
#include "LBPGPU.cuh"
//#include <opencv2\Blob_detection.hpp>
//#include <tbb\parallel_for.h>
//#include <tbb\blocked_range.h>

//Some constants for the algorithm
const double pi = 3.142;
const double cthr = 0.00001;
const double alpha = 0.002;
const double cT = 0.05;
const double covariance0 = 11.0;
const double cf = 0.1;
const double cfbar = 1.0 - cf;
const double temp_thr = 9.0*covariance0*covariance0;
const double prune = -alpha * cT;
const double alpha_bar = 1.0 - alpha;

void main()
{
	int i, j, k;
	i = j = k = 0;

	// Declare matrices to store original and resultant binary image
	cv::Mat orig_img;
	// Declare VideoCapture object for storing video

	std::stringstream ss;
	std::string path = "F:\\Afstuderen\\Afstuderen\\Videos\\";
	std::string file = "ss1_sc.avi";
	ss << path << file;
	std::string s = ss.str();
	cv::VideoCapture capture(s);
	capture.read(orig_img);

	/*cv::Rect Radar_scr = cv::Rect(135, 30, 180, 200);
	cv::Mat radar_img = orig_img(Radar_scr);
	
	std::vector<cv::Mat> channels_rad;
	cv::split(radar_img, channels_rad);
	
	imshow("Radar Image", radar_img);
	cv::threshold(channels_rad[2], radar_img, 120, 255,CV_THRESH_BINARY);*/
	
	//std::cout << "ret (python)  = " << std::endl << format(radar_img, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
	//std::vector<std::vector<cv::Point> > contours_rad;
	//std::vector<cv::Vec4i> hierarchy_rad;
	//findContours(radar_img, contours_rad, hierarchy_rad, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	//std::vector<std::vector<cv::Point> > contours_poly(contours_rad.size());
	//std::vector<cv::Rect> boundRect_rad(contours_rad.size());

	//for (int i = 0; i < contours_rad.size(); i++)
	//{
	//	//approxPolyDP(cv::Mat(contours_rad[i]), contours_poly[i], 3, true);
	//	boundRect_rad[i] = cv::boundingRect(contours_rad[i]);
	//}

	////Radar location
	//int own_loc[2] = { 89,98 };
	//int tar_loc[2] = {own_loc[0] - boundRect_rad[2].x ,own_loc[1]- boundRect_rad[2].y};
	//std::cout << "x = " << tar_loc[0] << ", y = " << tar_loc[1] << std::endl;
	//imshow("Radar",radar_img);

	//cv::waitKey(30);
	/*Timing*/
	double duration, duration1, duration2, duration3;
	duration1 = static_cast<double>(cv::getTickCount());

	//GPU objects
	cv::cuda::GpuMat frame, fgmask, fgimg;
	frame.upload(orig_img);
	fgimg.create(frame.size(), frame.type());

	cv::Ptr<cv::cuda::BackgroundSubtractorMOG> mog = cv::cuda::createBackgroundSubtractorMOG(80,5,0.9);
	cv::cuda::GpuMat bgimg;
	fgimg.setTo(cv::Scalar::all(0));
	cv::VideoWriter video("TurnSaliency_CovTrh.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, orig_img.size(), true);

	float w = (float)orig_img.cols, h = (float)orig_img.rows;
	float maxD = max(w, h);
	float max_dimension = 1000;

	while (1)
	{
		
		if (!capture.read(orig_img)) {
			break;
			capture.release();
			capture = cv::VideoCapture("Videoturncropped.mp4");
			capture.read(orig_img);
		}

		
		//Create LBP feature
		if (false) {
			Mat img;
			cvtColor(orig_img, img, CV_BGR2GRAY);
			medianBlur(img, img, 11);
			img.convertTo(img, CV_64F);

			lbp::LBP lbp(4, lbp::LBP_MAPPING_NONE);
			lbp.calcLBP(img, 1);
			Mat lbpImg = lbp.getLBPImage();
			imshow("orig_img", lbpImg);
			Mat lbp_img;
			resize(lbpImg, lbp_img, orig_img.size(), 0.0, 0.0, INTER_AREA);
			//Create feature matrix
			vector<Mat> channels;
			split(orig_img, channels);
			channels.push_back(lbp_img);
			merge(channels, orig_img);

			imshow("merge", orig_img);
			waitKey(0);
		}
	
		// Upload to GPU

		
		resize(orig_img, orig_img, Size((int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD)), 0.0, 0.0, INTER_AREA);
		duration = static_cast<double>(cv::getTickCount());
		Mat mean1, mean2, mean3, std1, std2, std3;
		vector<Mat> channels;
		split(orig_img, channels);
		meanStdDev(channels[0], mean1, std1);
		meanStdDev(channels[1], mean2, std2);
		meanStdDev(channels[2], mean3, std3);

	
		//loopcount++;
		//trh = mean1.at<double>(0) + 3 * std1.at<double>(0);
		double dist = 1.5;
		Mat bm1 = (channels[0] < mean1- dist*std1) | (channels[0] > mean1 + dist * std1);
		Mat bm2 = (channels[1] < mean2 - dist * std2) | (channels[1] > mean2 + dist* std2);
		Mat bm3 = (channels[2] < mean3 - dist * std3) | (channels[2] > mean3 + dist * std3);
	
		imshow("bm1", bm1);
		imshow("bm2", bm2);
		imshow("bm3", bm3);
		Mat res = bm1 + bm2 + bm3;
		normalize(res, res, 255, 0, NORM_MINMAX);
		imshow("res", res);


		frame.upload(orig_img);
		

		cv::cuda::cvtColor(frame, frame, CV_BGR2Lab);

		std::vector<cv::cuda::GpuMat> featureMaps;

		//Whiten feature map
		cv::cuda::split(frame, featureMaps);
			for (int i = 0; i < featureMaps.size(); i++)
			{
				cv::cuda::normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, cv::NORM_MINMAX, -1);
			}
		cv::cuda::merge(featureMaps, frame);

		//Initialize result
		fgimg.setTo(cv::Scalar::all(0));
		mog->apply(frame, fgmask);
		frame.copyTo(fgimg, fgmask);

		mog->getBackgroundImage(bgimg);

		cv::Ptr<cv::cuda::Filter> filterGaus = cv::cuda::createGaussianFilter(fgmask.type(), fgmask.type(), { 3,3 }, -1);
		filterGaus->apply(fgmask,fgmask);

		//Download result from Gpu
		cv::Mat result;
		fgmask.download(result);
		
		// Radar Detection
		//cv::Mat radar_img = orig_img(Radar_scr);
	
		////std::vector<cv::Mat> channels_rad;
		//cv::split(radar_img, channels_rad);
	
  // 		cv::Mat radar_thr;
		//cv::threshold(channels_rad[2], radar_thr, 120, 255,CV_THRESH_BINARY);
		//imshow("Radar_trh", radar_thr);
		//waitKey(40);
		////std::cout << "ret (python)  = " << std::endl << format(radar_img, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
		//std::vector<std::vector<cv::Point> > contours_rad;
		//std::vector<cv::Vec4i> hierarchy_rad;
		//findContours(radar_thr, contours_rad, hierarchy_rad, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		////std::vector<std::vector<cv::Point> > contours_poly(contours.size());
		//std::vector<cv::Rect> boundRect_rad;// (contours_rad.size());
		//									//Radar location
		//int own_loc[2] = { 89,98 };

		//for (int i = 0; i < contours_rad.size(); i++)
		//{
		//	//approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		//	//boundRect_rad[i] = cv::boundingRect(contours_rad[i]);
		//	boundRect_rad.push_back(cv::boundingRect(contours_rad[i]));
		//	if (i >= 2) {
		//		int tar_loc[2] = { own_loc[0] - boundRect_rad[i].x ,own_loc[1] - boundRect_rad[i].y };
		//		std::cout << ", x = " << tar_loc[0] << ", y = " << tar_loc[1] << std::endl;
		//	}
		//}






		//Contour locator
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours(result, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		
		double minArea = 10;
		std::vector<cv::Rect> savedBoundBox;

		std::vector<std::vector<cv::Point> > contours_poly(contours.size());// contours.size());
		std::vector<cv::Rect> boundRect(contours.size());// contours.size());

		for (int i = 0; i< contours.size(); i++)
		{			
			approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
			//int y = boundRect[i].y + boundRect[i].height / 2;

			//minArea = 200 - 0.23*y;
			double area = contourArea(contours[i]);
			if ((area > minArea))// && (frame.rows - boundRect[i].y + boundRect[i].height / 2 > 100))
			{
				savedBoundBox.push_back(boundRect[i]);
			}
		}

		/*double maxArea = 0.0
		if ((savedBoundBox.size() == 0) && (contours.size() != 0))
		{
			for (int i = 0; i< contours.size(); i++)
			{
				double area = contourArea(contours[i]);
				if (area > maxArea)
				{
					maxArea = area;
					savedContour.push_back(i);
				}
			}
		}*/


		// Draw bonding rects 
		cv::Mat drawing = cv::Mat::zeros(result.size(), CV_8UC3);
		
		cv::Scalar color = cv::Scalar(0, 200, 50);

		for (int i = 0; i < savedBoundBox.size(); i++)
		{
			rectangle(orig_img, savedBoundBox[i].tl(), savedBoundBox[i].br(), color);
		}

		//cv::waitKey(30);
		cv::imshow("video", orig_img);
		//cv::imshow("res", result);
		

		if (false) {

			video.write(orig_img);
		}

		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();

		std::cout << "\n duration :" << duration;
		std::cout << "  FPS: " << 1 / duration;
		//cv::waitKey(50);
		if (cv::waitKey(1) > 0)
			break;
	}

	duration1 = static_cast<double>(cv::getTickCount()) - duration1;
	duration1 /= cv::getTickFrequency();

	std::cout << "\n duration1 :" << duration1;

	video.release();
	_getch();
}



