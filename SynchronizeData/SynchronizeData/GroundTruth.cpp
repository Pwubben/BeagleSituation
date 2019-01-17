#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <vector>
#include <conio.h>
#include "opencv2/opencv.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "RadarScreenDetect.h"

using namespace cv;
using namespace std;

void GroundTruth(cv::VideoCapture capture,vector<Rect> &boundRectVec) {
	Mat data;
	capture.read(data);
	//imshow("data", data);

	// Detection of radar image and sea image
 	cv::Rect sea_scr;
	cv::Rect radar_scr;
	RadarScreenDetect(data, radar_scr, sea_scr);

	cv::Mat radar_src = data(radar_scr);
	//data = data(radar_scr);
	

	cv::Mat src;
	src = data(sea_scr);
	int count = 0;

	while (1)
	{
		capture >> src;

		if (src.empty())
		{
			// Reach end of the video file
			break;
		}

		src = src(sea_scr);
		imshow("true", src);
		//imshow("src", src);
		Mat src_hsv;
		cvtColor(src, src_hsv, cv::COLOR_BGR2HSV);

		// Setup ranges
		//Scalar low(36, 90, 90);
		//Scalar high(70, 255, 255);
		Scalar low(36, 60, 90);
		Scalar high(90, 255, 255);
		// Get binary mask
		Mat1b mask;
		inRange(src_hsv, low, high, mask);
		imshow("mask", mask);

		Mat invMask;
		bitwise_not(mask, invMask);
		vector<Point> zero_pts;
		findNonZero(invMask, zero_pts);

		vector<Point2f> pointsForSearch;
		for (int i = 0; i < zero_pts.size(); i++) {
			pointsForSearch.push_back((Point2f)zero_pts[i]);
		}

		flann::KDTreeIndexParams indexParams;
		flann::Index kdtree(Mat(pointsForSearch).reshape(1), indexParams);
		

		vector<Point> pts;
		findNonZero(mask, pts);
		Vec3b black(0, 0, 0);
		for (int i = 0; i < pts.size(); i++) {
			vector<float> query;
			query.push_back(pts[i].x);
			query.push_back(pts[i].y);
			vector<int> indices;
			vector<float> dists;
			kdtree.knnSearch(query, indices, dists, 1);

			
			//src.at<Vec3b>(pts[i]);
			src.at<Vec3b>(pts[i]) = src.at<Vec3b>(pointsForSearch[indices[0]]);
		}

		

		//// Define the radius tolerance
		//int th_distance = 50; // radius tolerance

		//					  // Apply partition 
		//					  // All pixels within the radius tolerance distance will belong to the same class (same label)
		//vector<int> labels;

		//// With lambda function 
		//int th2 = th_distance * th_distance;
		//int n_labels = partition(pts, labels, [th2](const Point& lhs, const Point& rhs) {
		//	return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < th2;
		//});

		//// You can save all points in the same class in a vector (one for each class), just like findContours
		//vector<vector<Point>> contours(n_labels);
		//for (int i = 0; i < pts.size(); ++i)
		//{
		//	contours[labels[i]].push_back(pts[i]);
		//}

		//// Get bounding boxes
		//vector<Rect> boxes;
		//for (int i = 0; i < contours.size(); ++i)
		//{
		//	Rect box = boundingRect(contours[i]);
		//	boxes.push_back(box);
		//}

		//Rect largest_box;
		//if (boxes.size() > 0) {
		//	largest_box = *max_element(boxes.begin(), boxes.end(), [](const Rect& lhs, const Rect& rhs) {return lhs.area() < rhs.area(); });
		//}
		////boundRect = boundingRect(contours);

		//boundRectVec.push_back(largest_box);

		//// Draw bonding rects 
		//Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
		//RNG rng(0xFFFFFFFF);
		//Scalar color = Scalar(0, 0, 200);

		//rectangle(src, largest_box.tl(), largest_box.br(), color);
		/*for (int i = 0; i < contours.size(); i++)
		{
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color);
		}*/
		imshow("src_draw", src);
		
		if (count == 0) {
			for (int i = 0; i < 800; i++) {
				capture >> src;
				count++;
			}
		}
		//if (count == 10)
		//	break;

		if (cv::waitKey(1) > 0)
			break;
	}

}