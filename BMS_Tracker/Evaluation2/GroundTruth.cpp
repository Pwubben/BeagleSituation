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

void DataGeneration(std::string videoFile, std::string groundTruthFile, std::string avgTimeFile, std::string ResultFile, int GT_offset,int stopFrame) {
	
	cout << videoFile << endl;
	vector<vector<int>> GroundTruth;
	vector<Rect> GT;
	readGroundTruth(getFileString(groundTruthFile), GroundTruth);
	for (int s = 0; s < GroundTruth.size(); s++) {
		Rect coord(GroundTruth[s][0], GroundTruth[s][1], GroundTruth[s][2], GroundTruth[s][3]);
		GT.push_back(coord);
	}

	// Declare VideoCapture object for storing video
	cv::VideoCapture capture(getFileString(videoFile));

	//Output parameters of lgorithms
	double avg_timeSaliency = 0.0, avg_timeGMM = 0.0;

	//Output storage vectors over multiple parameter settings
	vector<double> avg_timeSaliencyData, avg_timeGMMData;

	bool check(false);
	
	try {
		if (!check) {
			//Run for all desired parameter combinations
			double max_dimension = 800;
			double sample_step = 25; 
			double threshold = 5;
			//Clear data
			capture.release();

			cv::VideoCapture capture(getFileString(videoFile));
		
			SaliencyDetect(capture, avg_timeSaliency, max_dimension, sample_step, threshold, GT, GT_offset,stopFrame);
		}
		else {
			check = false;
		}
	}
	catch (std::exception e) {
		check = true;
		std::cout << e.what() << std::endl;
	}
	//std::cout << "ret (python)  = " << std::endl << format(data, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

}

//void GroundTruth(cv::VideoCapture capture,vector<Rect> &boundRectVec) {
//	Mat data;
//	capture.read(data);
//	//imshow("data", data);
//
//	// Detection of radar image and sea image
// 	cv::Rect sea_scr;
//	cv::Rect radar_scr;
//	RadarScreenDetect(data, radar_scr, sea_scr);
//
//	cv::Mat radar_src = data(radar_scr);
//	//data = data(radar_scr);
//	
//
//	cv::Mat src;
//	src = data(sea_scr);
//	int count = 0;
//
//	while (1)
//	{
//		capture >> src;
//
//		if (src.empty())
//		{
//			// Reach end of the video file
//			break;
//		}
//
//		src = src(sea_scr);
//		//imshow("src", src);
//		Mat src_hsv;
//		cvtColor(src, src_hsv, CV_BGR2HSV);
//
//		// Setup ranges
//		Scalar low(36, 90, 90);
//		Scalar high(70, 255, 255);
//
//		// Get binary mask
//		Mat1b mask;
//		inRange(src_hsv, low, high, mask);
//		//imshow("mask", mask);
//
//		vector<Point> pts;
//		findNonZero(mask, pts);
//
//		// Define the radius tolerance
//		int th_distance = 50; // radius tolerance
//
//							  // Apply partition 
//							  // All pixels within the radius tolerance distance will belong to the same class (same label)
//		vector<int> labels;
//
//		// With lambda function 
//		int th2 = th_distance * th_distance;
//		int n_labels = partition(pts, labels, [th2](const Point& lhs, const Point& rhs) {
//			return ((lhs.x - rhs.x)*(lhs.x - rhs.x) + (lhs.y - rhs.y)*(lhs.y - rhs.y)) < th2;
//		});
//
//		// You can save all points in the same class in a vector (one for each class), just like findContours
//		vector<vector<Point>> contours(n_labels);
//		for (int i = 0; i < pts.size(); ++i)
//		{
//			contours[labels[i]].push_back(pts[i]);
//		}
//
//		// Get bounding boxes
//		vector<Rect> boxes;
//		for (int i = 0; i < contours.size(); ++i)
//		{
//			Rect box = boundingRect(contours[i]);
//			boxes.push_back(box);
//		}
//
//		Rect largest_box;
//		if (boxes.size() > 0) {
//			largest_box = *max_element(boxes.begin(), boxes.end(), [](const Rect& lhs, const Rect& rhs) {return lhs.area() < rhs.area(); });
//		}
//		//boundRect = boundingRect(contours);
//
//		boundRectVec.push_back(largest_box);
//
//		// Draw bonding rects 
//		//Mat drawing = Mat::zeros(mask.size(), CV_8UC3);
//		//RNG rng(0xFFFFFFFF);
//		//Scalar color = Scalar(0, 0, 200);
//
//		//rectangle(src, largest_box.tl(), largest_box.br(), color);
//		/*for (int i = 0; i < contours.size(); i++)
//		{
//			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color);
//		}*/
//		//imshow("src_draw", src);
//		count++;
//		//if (count == 10)
//		//	break;
//
//		if (cv::waitKey(1) > 0)
//			break;
//	}
//
//
//
//}

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


void writeResultFile(vector<vector<int>> falsePositiveCount, vector<vector<int>> truePositiveCount,vector<vector<double>> precision, vector<double> recall, vector<vector<double>> IoU, std::ofstream &File) {
	//Parameter 
	for (int n = 0; n < falsePositiveCount.size(); n++) {
		//Time 
		for (int k = 0; k < falsePositiveCount[n].size(); k++) {
				File << falsePositiveCount[n][k] << "," << truePositiveCount[n][k] << "," << precision[n][k] << "," << IoU[n][k] << endl;
		}
		File << "NP" << "," << recall[n]<< endl;
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

void writeFileNames(std::string File, ::string& videoFile, std::string& boundRectFileSal, std::string& avgTimeFileSal, std::string& boundRectFileGMM, std::string& avgTimeFileGMM, std::string& labelFile) {
	std::stringstream ss;
	ss << File << ".avi";
	videoFile = ss.str();
	ss.clear();
}

//Storage

//void HorizonDetect(Mat src) {
//	Mat src_gray;
//	Mat left_vec, right_vec;
//	Point maxL, maxR, minL, minR;
//	double maxValL, minValL, maxValR, minValR;
//
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//	GaussianBlur(src_gray, src_gray, { 5,5 }, 6);
//	imshow("srcgray", src_gray);
//	Sobel(src_gray(Range::all(), Range(0, 8)), left_vec, CV_32F, 0, 1, 9);
//	Sobel(src_gray(Range::all(), Range(src_gray.cols - 12, src_gray.cols - 4)), right_vec, CV_32F, 0, 1, 9);
//
//	cv::minMaxLoc(left_vec, &minValL, &maxValL, &minL, &maxL);
//	cv::minMaxLoc(right_vec, &minValR, &maxValR, &minR, &maxR);
//	Point HorL = minL;
//	Point HorR = minR;
//	if (-minValL < maxValL) {
//		HorL = maxL;
//	}
//	if (-minValR < maxValR) {
//		HorR = maxR;
//	}
//	HorR.x = src_gray.cols - 3;
//
//	std::vector<Mat> channels;
//	split(src, channels);
//	Mat dst;
//	Mat flag_mask = (channels[2] < 200);
//	imshow("flag", flag_mask);
//	channels[2].copyTo(dst, flag_mask);
//	channels.erase(channels.begin() + 2);
//	channels.push_back(dst);
//	merge(channels, src);
//	imshow("red", dst);
//}