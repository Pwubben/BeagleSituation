// GMM_IMG.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <vector>
#include <conio.h>
#include "opencv2/ml.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace ml;
int main()
{
	cv::Mat src;
	cv::VideoCapture capture("TestVid.mp4");
	capture.read(src);
	vector<Rect> GroundTruth;
	while (1)
	{
		capture >> src;
		
		cvtColor(src, src, CV_BGR2GRAY);


		Ptr<ml::EM> em = ml::EM::create();
		em->setClusterNumber(5);

		em->trainEM(src);

		//vector<vector<Point> > contours;
		//vector<Vec4i> hierarchy;
		//cv::findContours(mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		//vector<vector<Point> > contours_poly(contours.size());
		//vector<Rect> boundRect(contours.size());

		//for (int i = 0; i < contours.size(); i++)
		//{
		//	approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		//	boundRect[i] = boundingRect(Mat(contours_poly[i]));
		//}

		//// Draw bonding rects 
		//Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
		//RNG rng(0xFFFFFFFF);
		//Scalar color = Scalar(0, 0, 200);

		//for (int i = 0; i < contours.size(); i++)
		//{
		//	rectangle(src, boundRect[i].tl(), boundRect[i].br(), color);
		//}
		//imshow("src_draw", src);


		//GroundTruth.push_back(boundRect[0]);

		if (cv::waitKey(30) > 0)
			break;
	}

	return 0;
}

