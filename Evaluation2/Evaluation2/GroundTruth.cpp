#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <vector>
#include <conio.h>
#include <numeric>
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
	if (l1.y > r2.y || l2.y > r1.y)
		return false;

	return true;
}

void trueFalsePositiveRate(vector<vector<vector<vector<int>>>> boundRectData, vector<vector<int>> groundTruth, vector<vector<int>> &falsePositiveCount, vector<vector<int>> &truePositiveCount, 
	vector<vector<double>> &precision, vector<double> &recall,vector<vector<double>> &IoU) {

	vector<vector<int>> falseNegativeCount;

	//Parameter
	for (int i = 0; i < boundRectData.size(); i++) {
		vector<double> IoUtime(0.0);
		vector<double> precisionTime(0.0);
		//vector<double> recallTime(0.0);
		falsePositiveCount.push_back(vector<int>(0));
		falseNegativeCount.push_back(vector<int>(0));
		truePositiveCount.push_back(vector<int>(0));
		int avgCount = 0;
		//Time
		for (int j = 0; j < boundRectData[i].size(); j++) {
			double IoUavg = 0.0;
			falsePositiveCount[i].push_back(0);
			falseNegativeCount[i].push_back(0);
			truePositiveCount[i].push_back(0);

			//Key points of ground truth rectangle
			cv::Point l2(groundTruth[j][0], groundTruth[j][1]);
			cv::Point r2(groundTruth[j][0] + groundTruth[j][2], groundTruth[j][1] + groundTruth[j][3]);

			//Bounding box
			for (int k = 0; k < boundRectData[i][j].size(); k++) {
				//Key points of bounding rectangles
				cv::Point l1(boundRectData[i][j][k][0], boundRectData[i][j][k][1]);
				cv::Point r1(boundRectData[i][j][k][0] + boundRectData[i][j][k][2], boundRectData[i][j][k][1] + boundRectData[i][j][k][3]);

				//Check for overlap resulting in false or true positive
				if (doOverlap(l1, r1, l2, r2)) {
					truePositiveCount[i][j]++;
					IoUavg = IntersectionOverUnion(l1, r1, l2, r2);
					avgCount++;
				}
				else {
					falsePositiveCount[i][j]++;
				}
			}

			// When present target is not correctly detected, increase false negative count
			if ((truePositiveCount[i][j] == 0) && (groundTruth[j][0] != 0) && (groundTruth[j][1] != 0)) {
				falseNegativeCount[i][j]++;
			}

			// Area over Union per time step
			IoUtime.push_back(IoUavg);
			precisionTime.push_back(truePositiveCount[i][j] / float(truePositiveCount[i][j] + falsePositiveCount[i][j]));
			//recallTime.push_back(truePositiveCount[i][j] / float(truePositiveCount[i][j] + falseNegativeCount[i][j]));
		}
		// Area over Union for parameter set
		IoU.push_back(IoUtime);
		precision.push_back(precisionTime);
		recall.push_back(std::accumulate(truePositiveCount[i].begin(), truePositiveCount[i].end(), 0)/float(std::accumulate(truePositiveCount[i].begin(), truePositiveCount[i].end(), 0)+ std::accumulate(falseNegativeCount[i].begin(), falseNegativeCount[i].end(), 0)));
	}
}

double IntersectionOverUnion(cv::Point l1,cv::Point r1,cv::Point l2,cv::Point r2) {

	int xA = max(l1.x, l2.x);
	int yA = max(l1.y, l2.y);
	int xB = min(r1.x, r2.x);
	int yB = min(r1.y, r2.y);

	//Compute the area of intersection rectangle
	double interArea = max(0, xB - xA) * max(0, yB - yA);

	//Compute the area of both the prediction and ground - truth
	//Rectangles
	double boxAArea = (r1.x - l1.x) * (r1.y - l1.y);
	double boxBArea = (r2.x - l2.x) * (r2.y - l2.y);
		

	//Compute the intersection over union by taking the intersection
	//area and dividing it by the sum of prediction + ground - truth
	//areas - the interesection area
	double IoU = interArea / float(boxAArea + boxBArea - interArea);
	return IoU;
}

void writeBoundRectFile(vector<vector<vector<Rect>>> boundRectData, std::ofstream &File) {
	//Parameter 
	for (int n = 0; n < boundRectData.size(); n++) {
		//Time 
		for (int k = 0; k < boundRectData[n].size(); k++) {
			//Bounding box 
			for (int l = 0; l < boundRectData[n][k].size(); l++) {
				File << boundRectData[n][k][l].x << "," << boundRectData[n][k][l].y << "," << boundRectData[n][k][l].width << "," << boundRectData[n][k][l].height << endl;
			}
			File << "NT" << endl;
		}
		File << "NP" << endl;
	}
	File.close();
}

std::string getFileString(std::string fileName) {
	std::string path = "F:\\Afstuderen\\";
	std::stringstream ss;
	ss << path << fileName;
	std::string file = ss.str();
	return file;
}
//void writeEvaluationFile(vector<vector<vector<Rect>>> boundRectData, std::ofstream &File)