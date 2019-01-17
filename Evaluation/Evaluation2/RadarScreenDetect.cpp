#include "stdafx.h"
#include <iostream>
#include <set>

#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <ctime>
#include <string>
#include <vector>
#include <omp.h>


void RadarScreenDetect(cv::Mat src, cv::Rect& radar_scr, cv::Rect& sea_scr) {
	cv::Mat src_gray;
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
	GaussianBlur(src_gray, src_gray, cv::Size(9, 9), 2, 2);


	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 1000, 100, 50);
	
	/// Draw the circles detected

	cv::Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
	int radius = cvRound(circles[0][2]);
	// circle center
	circle(src, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
	// circle outline
	circle(src, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
	
	int radarMin = center.y - radius;
	if (radarMin < 0) {
		radarMin = 0;}

	radar_scr = cv::Rect(center.x - radius, radarMin, 2 * radius, 2 * radius);
	sea_scr = cv::Rect(10, center.y + radius+70, src.cols-10, src.rows-center.y-radius-70);
	//cv::imshow("Hough", src);
	//cv::waitKey(0);

}