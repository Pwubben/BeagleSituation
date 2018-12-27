#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <vector>
#include <conio.h>
#include "opencv2/opencv.hpp"
#include "RadarScreenDetect.h"

using namespace cv;
using namespace std;

void GroundTruth(cv::VideoCapture capture,vector<vector<Rect>> boundRectVec) {
	Mat data;
	capture.read(data);

	// Detection of radar image and sea image
	cv::Rect sea_scr;
	cv::Rect radar_scr;
	RadarScreenDetect(data, radar_scr, sea_scr);

	cv::Mat radar_src = data(radar_scr);
	//data = data(radar_scr);


	cv::Mat src;
	cv::VideoCapture capture("TestVid.mp4");
	capture.read(src);
	vector<Rect> GroundTruth;
	while (1)
	{
		capture >> src;
		imshow("src", src);
		Mat src_hsv;
		cvtColor(src, src_hsv, CV_BGR2HSV);

		// Setup ranges
		Scalar low(36, 90, 90);
		Scalar high(70, 255, 255);

		// Get binary mask
		Mat1b mask;
		inRange(src_hsv, low, high, mask);
		imshow("mask", mask);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}
		boundRectVec.push_back(boundRect);

		// Draw bonding rects 
		Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
		RNG rng(0xFFFFFFFF);
		Scalar color = Scalar(0, 0, 200);

		for (int i = 0; i < contours.size(); i++)
		{
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color);
		}
		imshow("src_draw", src);


		GroundTruth.push_back(boundRect[0]);

		if (cv::waitKey(30) > 0)
			break;
	}
}