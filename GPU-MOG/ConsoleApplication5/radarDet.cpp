//#include "pch.h"
//#include<opencv2/opencv.hpp>
//#include <vector>
//#include<iostream>
//
//using namespace cv;
//void getRadarDetection(Mat orig_img, std::vector<int> & detections) {
//	int radar_width = 180, radar_height = 200;
//	cv::Rect Radar_scr = cv::Rect(135, 30, radar_width, radar_height);
//	cv::Mat radar_img = orig_img(Radar_scr);
//
//	std::vector<cv::Mat> channels_rad;
//	std::vector<std::vector<cv::Point> > contours_rad;
//
//	//std::vector<cv::Mat> channels_rad;
//	cv::split(radar_img, channels_rad);
//
//	cv::Mat radar_thr;
//	cv::threshold(channels_rad[2], radar_thr, 120, 255, CV_THRESH_BINARY);
//	imshow("Radar_trh", radar_thr);
//	waitKey(40);
//
//
//	std::vector<cv::Vec4i> hierarchy_rad;
//	findContours(radar_thr, contours_rad, hierarchy_rad, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
//
//	//std::vector<std::vector<cv::Point> > contours_poly(contours.size());
//	std::vector<cv::Rect> boundRect_rad;// (contours_rad.size());
//
//	//Radar location
//
//	int own_loc[2] = { 89,98 };
//	double min;
//	minMaxLoc()
//
//	for (int i = 0; i < contours_rad.size(); i++)
//	{
//		//approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
//		//boundRect_rad[i] = cv::boundingRect(contours_rad[i]);
//		boundRect_rad.push_back(cv::boundingRect(contours_rad[i]));
//
//		if (i >= 2) {
//			detections.push_back(own_loc[0] - boundRect_rad[i].x);
//			detections.push_back(own_loc[1] - boundRect_rad[i].y);
//			std::cout << ", x = " << detections[0] << ", y = " << detections[1] << std::endl;
//		}
//	}
//
//}