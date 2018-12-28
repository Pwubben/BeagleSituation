#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <vector>
#include <conio.h>
#include "opencv2/opencv.hpp"
#include "RadarScreenDetect.h"
#include "DetectAlgorithms.h"

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
		//imshow("src", src);
		Mat src_hsv;
		cvtColor(src, src_hsv, CV_BGR2HSV);

		// Setup ranges
		Scalar low(36, 90, 90);
		Scalar high(70, 255, 255);

		// Get binary mask
		Mat1b mask;
		inRange(src_hsv, low, high, mask);
		//imshow("mask", mask);

		vector<Point> pts;
		findNonZero(mask, pts);

		// Define the radius tolerance
		int th_distance = 50; // radius tolerance

							  // Apply partition 
							  // All pixels within the radius tolerance distance will belong to the same class (same label)
		vector<int> labels;

		// With lambda function 
		int th2 = th_distance * th_distance;
		int n_labels = partition(pts, labels, [th2](const Point& lhs, const Point& rhs) {
			return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < th2;
		});

		// You can save all points in the same class in a vector (one for each class), just like findContours
		vector<vector<Point>> contours(n_labels);
		for (int i = 0; i < pts.size(); ++i)
		{
			contours[labels[i]].push_back(pts[i]);
		}

		// Get bounding boxes
		vector<Rect> boxes;
		for (int i = 0; i < contours.size(); ++i)
		{
			Rect box = boundingRect(contours[i]);
			boxes.push_back(box);
		}

		Rect largest_box;
		if (boxes.size() > 0) {
			largest_box = *max_element(boxes.begin(), boxes.end(), [](const Rect& lhs, const Rect& rhs) {return lhs.area() < rhs.area(); });
		}
		//boundRect = boundingRect(contours);

		boundRectVec.push_back(largest_box);

		// Draw bonding rects 
		//Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
		//RNG rng(0xFFFFFFFF);
		//Scalar color = Scalar(0, 0, 200);

		//rectangle(src, largest_box.tl(), largest_box.br(), color);
		/*for (int i = 0; i < contours.size(); i++)
		{
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color);
		}*/
		//imshow("src_draw", src);
		count++;
		if (count == 10)
			break;

		if (cv::waitKey(1) > 0)
			break;
	}



}

void readGroundTruth(std::string fileName, vector<vector<int>>& groundTruth)
{
	ifstream file(fileName);
	string line;
	while (getline(file, line))
	{
		vector<int> row;
		stringstream iss(line);
		string val;

		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			row.push_back(stoi(val));
		}
		groundTruth.push_back(row);
	}
}

void readBoundRectData(std::string fileName, vector<vector<vector<vector<int>>>> &boundRectData)
{
	ifstream file(fileName);
	string line;
	int parameterCount = 0;
	int timeCount = 0;
	int boundRectCount = 0;

	vector<vector<int>> boundRectVec;
	vector<vector<vector<int>>> timeVec;

	while (getline(file, line))
	{
		boundRectCount = 0;
		vector<int> boundRect;
		stringstream iss(line);
		string val;
	
		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			if (val == "NT") {
				timeCount++;
				timeVec.push_back(boundRectVec);
				boundRectVec.clear();
				break;
			}
			if (val == "NP") {
				parameterCount++;
				boundRectData.push_back(timeVec);
				timeVec.clear();
				break;
			}
			else {
				boundRect.push_back(stoi(val));
			}
		}
		if (boundRect.empty()) {
			continue;
		}
		boundRectVec.push_back(boundRect);
		boundRectCount++;
	}
}

bool doOverlap(Point l1, Point r1, Point l2, Point r2)
{
	// If one rectangle is on left side of other 
	if (l1.x > r2.x || l2.x > r1.x)
		return false;

	// If one rectangle is above other 
	if (l1.y < r2.y || l2.y < r1.y)
		return false;

	return true;
}