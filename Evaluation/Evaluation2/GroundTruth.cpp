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

void DataGeneration(std::string videoFile, std::string groundTruthFile, std::string SalResultFile, std::string GMMResultFile, std::string boundRectFileGMM, std::string avgTimeFileGMM, std::string labelFile, int GT_offset,int stopFrame) {
	
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
	vector<vector<Rect>> boundRectSaliency, boundRectGMM;
	double avg_timeSaliency = 0.0, avg_timeGMM = 0.0;

	//Output storage vectors over multiple parameter settings
	vector<vector<vector<Rect>>> boundRectSaliencyData, boundRectGMMData;
	vector<double> avg_timeSaliencyData, avg_timeGMMData;

	//Matrix for storing parameter combinations
	Mat data = cv::Mat::zeros(400, 6, CV_64F);
	int cycleCountSal = 0;
	int cycleCountGMM = 0;

	vector<vector<double>> falsePositiveCountGMM, truePositiveCountGMM, trueNegativeCountGMM, falseNegativeCountGMM, falsePositiveAreaGMM, truePositiveAreaGMM, precisionAreaGMM, IoUGMM;
	vector<vector<double>> falsePositiveCountSal, truePositiveCountSal, precisionCountSal, falsePositiveAreaSal, truePositiveAreaSal, precisionAreaSal, IoUSal;
	
	bool check(false);
	
	try {
		if (!check) {
			//Run for all desired parameter combinations
			for (double max_dimension = 1600; max_dimension < 1601; max_dimension += 400) {
				for (double sample_step = 25; sample_step < 26; sample_step += 15) {

					//Clear data
					boundRectSaliency.clear();
					capture.release();

					cv::VideoCapture capture(getFileString(videoFile));
					//Parameter documentation
					data.at<double>(cycleCountSal, 0) = max_dimension;
					data.at<double>(cycleCountSal, 1) = sample_step;


					SaliencyDetect(capture, boundRectSaliency, avg_timeSaliency, max_dimension, sample_step, 3, GT, GT_offset, falsePositiveCountSal, truePositiveCountSal, precisionCountSal, falsePositiveAreaSal, truePositiveAreaSal, precisionAreaSal, IoUSal);

					boundRectSaliencyData.push_back(boundRectSaliency);
					avg_timeSaliencyData.push_back(avg_timeSaliency);
					cycleCountSal++;

				}
				for (double backGroundRatio = 600; backGroundRatio < 901; backGroundRatio += 6) {
					for (double timeHorizon = 200; timeHorizon < 201; timeHorizon += 60) {
						//Push back result vectors for new threshold value
						falsePositiveCountGMM.push_back(vector<double>());
						truePositiveCountGMM.push_back(vector<double>());
						trueNegativeCountGMM.push_back(vector<double>());
						falseNegativeCountGMM.push_back(vector<double>());
						falsePositiveAreaGMM.push_back(vector<double>());
						truePositiveAreaGMM.push_back(vector<double>());
						precisionAreaGMM.push_back(vector<double>());
						IoUGMM.push_back(vector<double>());
						//Clear data
						boundRectGMM.clear();
						capture.release();

						cv::VideoCapture capture(getFileString(videoFile));

						data.at<double>(cycleCountGMM, 2) = max_dimension;
						data.at<double>(cycleCountGMM, 3) = backGroundRatio;
						data.at<double>(cycleCountGMM, 4) = timeHorizon;

						GMMDetect(capture, boundRectGMM, avg_timeGMM, max_dimension, backGroundRatio, timeHorizon, GT, GT_offset, falsePositiveCountGMM, truePositiveCountGMM, trueNegativeCountGMM, falseNegativeCountGMM, falsePositiveAreaGMM, truePositiveAreaGMM, precisionAreaGMM, IoUGMM,stopFrame);
						boundRectGMMData.push_back(boundRectGMM);
						avg_timeGMMData.push_back(avg_timeGMM);

						cycleCountGMM++;
					}
				}
			}
		}
		else {
			check = false;
		}
	}
	catch (std::exception e) {
		check = true;
		std::cout << e.what() << std::endl;
	}
	std::cout << "ret (python)  = " << std::endl << format(data, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

	//Take averages 
	vector<double> meanfalsePositiveCountGMM, meantruePositiveCountGMM, meanTrueNegativeCountGMM,meanFalseNegativeCountGMM, meanfalsePositiveAreaGMM, meantruePositiveAreaGMM, meanprecisionAreaGMM, meanIoUGMM;
	for (int i = 0; i < falsePositiveCountGMM.size(); i++) {
		meanfalsePositiveCountGMM.push_back(accumulate(falsePositiveCountGMM[i].begin(), falsePositiveCountGMM[i].end(), 0.0));
		meantruePositiveCountGMM.push_back(accumulate(truePositiveCountGMM[i].begin(), truePositiveCountGMM[i].end(), 0.0) );
		meanTrueNegativeCountGMM.push_back(accumulate(trueNegativeCountGMM[i].begin(), trueNegativeCountGMM[i].end(), 0.0) );
		meanFalseNegativeCountGMM.push_back(accumulate(falseNegativeCountGMM[i].begin(), falseNegativeCountGMM[i].end(), 0.0) );

		meanfalsePositiveAreaGMM.push_back(accumulate(falsePositiveAreaGMM[i].begin(), falsePositiveAreaGMM[i].end(), 0.0) / falsePositiveAreaGMM[i].size());
		meantruePositiveAreaGMM.push_back(accumulate(truePositiveAreaGMM[i].begin(), truePositiveAreaGMM[i].end(), 0.0) / truePositiveAreaGMM[i].size());
		meanprecisionAreaGMM.push_back(accumulate(precisionAreaGMM[i].begin(), precisionAreaGMM[i].end(), 0.0) / precisionAreaGMM[i].size());
		meanIoUGMM.push_back(accumulate(IoUGMM[i].begin(), IoUGMM[i].end(), 0.0) / IoUGMM[i].size());
	}

	//Compute performance result
	vector<double> TPR, FPR, precision;
	for (int i = 0; i < meanfalsePositiveCountGMM.size(); i++) {
		TPR.push_back(double(meantruePositiveCountGMM[i] / float(meantruePositiveCountGMM[i] + meanFalseNegativeCountGMM[i])));
		FPR.push_back(double(meanfalsePositiveCountGMM[i] / float(meanTrueNegativeCountGMM[i] + meanfalsePositiveCountGMM[i])));
		precision.push_back(double(meantruePositiveCountGMM[i] / float(meantruePositiveCountGMM[i] + meanfalsePositiveCountGMM[i])));
	}
	//Saliecy Result File
	/*std::string file = getFileString(SalResultFile);

	ofstream SalResultFileName(file, std::ofstream::out | std::ofstream::trunc);

	for (int n = 0; n < falsePositiveCountSal.size(); n++){
		for (int j = 0; j < falsePositiveCountSal[n].size(); j++) {
			SalResultFileName << falsePositiveCountSal[n][j] << "," << truePositiveCountSal[n][j] << "," << precisionCountSal[n][j] << "," << falsePositiveAreaSal[n][j] << "," << truePositiveAreaSal[n][j] << "," << precisionAreaSal[n][j] << "," << IoUSal[n][j] << endl;
		}
		SalResultFileName << "NP" << endl;
	}
	SalResultFileName.close();*/

	//GMM Result file
	std::string file = getFileString(GMMResultFile);

	ofstream GMMResultFileName(file, std::ofstream::out | std::ofstream::trunc);
	int resultCount = 0;
	for (int n = 0; n<meanfalsePositiveCountGMM.size(); n++)
	{
		GMMResultFileName << FPR[n] << "," << TPR[n] << "," << precision[n] << ","<< meanfalsePositiveAreaGMM[n] << "," << meantruePositiveAreaGMM[n] << "," << meanprecisionAreaGMM[n] << "," << meanIoUGMM[n] << "," << meanfalsePositiveCountGMM[n] << "," << meantruePositiveCountGMM[n] << "," << meanTrueNegativeCountGMM[n] << "," << meanFalseNegativeCountGMM[n] <<endl;
	}
	GMMResultFileName.close();

	//Saliency
	////Bounding rect vector file write
	//std::string file = getFileString(boundRectFileSal);
	//std::ofstream ss3_scBoundRectSalFile(file, std::ofstream::out | std::ofstream::trunc);

	//writeBoundRectFile(boundRectSaliencyData, ss3_scBoundRectSalFile);

	//GMM 
	/*file = getFileString(avgTimeFileSal);

	ofstream avgtimeSalFile(file, std::ofstream::out | std::ofstream::trunc);

	for (int n = 0; n<avg_timeSaliencyData.size(); n++)
	{
		avgtimeSalFile << avg_timeSaliencyData[n] << endl;
	}
	avgtimeSalFile.close();*/

	//GMM
	//Bounding rect vector file write
	//std::string file = getFileString(boundRectFileGMM);
	//std::ofstream ss3_scBoundRectGMMFile(file, std::ofstream::out | std::ofstream::trunc);

	//writeBoundRectFile(boundRectGMMData, ss3_scBoundRectGMMFile);

	////Average time file write
	//file = getFileString(avgTimeFileGMM);

	//ofstream avgtimeGMMFile(file, std::ofstream::out | std::ofstream::trunc);
	//for (int n = 0; n<avg_timeGMMData.size(); n++)
	//{
	//	avgtimeGMMFile << avg_timeGMMData[n] << endl;
	//}
	//avgtimeGMMFile.close();

	//Parameter file write
	file = getFileString(labelFile);
	ofstream vectorlabelFile(file, std::ofstream::out | std::ofstream::trunc);

	vectorlabelFile << cv::format(data, cv::Formatter::FMT_CSV) << std::endl;
	vectorlabelFile.close();
}

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
		//if (count == 10)
		//	break;

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
	if (r2.x == 0 && r2.y == 0)
		return false;

	return true;
}

void trueFalsePositiveRateROCGMM(vector<Rect> boundRectData, Rect groundTruth, vector<vector<double>> &falsePositiveCount, vector<vector<double>> &truePositiveCount, vector<vector<double>> &trueNegativeCount,
	vector<vector<double>> &falseNegativeCount, vector<vector<double>> &falsePositiveArea, vector<vector<double>> &truePositiveArea,
	vector<vector<double>> &precisionArea, vector<vector<double>> &IoU, int GT_offset, int timeCount) {

	int falseNegatives = 0, truePositives = 0, falsePositives = 0, trueNegatives = 0;
	double IoUavg = 0;
	//Key points of ground truth rectangle
	cv::Point l2(groundTruth.x, groundTruth.y);
	cv::Point r2(groundTruth.x + groundTruth.width, groundTruth.y + groundTruth.height);
	double intersectArea = 0.0, falseArea = 0.0;

	for (int i = 0; i < boundRectData.size(); i++) {
		int count = 0;

		//Key points of bounding rectangles
		cv::Point l1(boundRectData[i].x, boundRectData[i].y);
		cv::Point r1(boundRectData[i].x + boundRectData[i].width, boundRectData[i].y + boundRectData[i].height);

		//Check for overlap resulting in false or true positive
		if (doOverlap(l1, r1, l2, r2)) {
			intersectArea += IntersectionArea(l1, r1, l2, r2);
			falseArea += nonIntersect(l1, r1, l2, r2);
			IoUavg += IntersectionOverUnion(l1, r1, l2, r2);
			if (truePositives == 0)
				truePositives++;
		}
		else {
			falseArea += boundRectData[i].width*boundRectData[i].height;
			falsePositives++;
		}
	}
	if ((truePositives == 0) && (groundTruth.x != 0) && (groundTruth.y != 0)) {
		falseNegatives++;
	}
	if ((truePositives == 0) && (groundTruth.x == 0) && (groundTruth.y == 0)) {
		trueNegatives++;
	}
	//Area evaluation 
	truePositiveArea.back().push_back(double(intersectArea / float(groundTruth.width*groundTruth.height)));
	falsePositiveArea.back().push_back(double(intersectArea / float(1902*536 -(intersectArea + falseArea))));
	precisionArea.back().push_back(double(intersectArea / float(intersectArea + falseArea)));

	//Count evaluation
	truePositiveCount.back().push_back(double(truePositives));
	falsePositiveCount.back().push_back(double(falsePositives));
	falseNegativeCount.back().push_back(double(falseNegatives));
	trueNegativeCount.back().push_back(double(trueNegatives));
	
	//IoU
	IoU.back().push_back(IoUavg);

	if (isnan(truePositiveArea.back().back()) == 1) {
		truePositiveArea.back().back() = 0;
	}
	if (isnan(falsePositiveArea.back().back()) == 1) {
		falsePositiveArea.back().back() = 0;
	}
	if (isnan(precisionArea.back().back()) == 1) {
		precisionArea.back().back() = 0;
	}
	if (isnan(truePositiveCount.back().back()) == 1) {
		truePositiveCount.back().back() = 0;
	}
	if (isnan(falsePositiveCount.back().back()) == 1) {
		falsePositiveCount.back().back() = 0;
	}
	if (isnan(IoU.back().back()) == 1) {
		IoU.back().back() = 0;
	}
}

void trueFalsePositiveRateROC(vector<Rect> boundRectData, Rect groundTruth, vector<vector<double>> &falsePositiveCount, vector<vector<double>> &truePositiveCount,
	vector<vector<double>> &precisionCount, vector<vector<double>> &falsePositiveArea, vector<vector<double>> &truePositiveArea,
	vector<vector<double>> &precisionArea, vector<vector<double>> &IoU, int GT_offset, int timeCount) {

	int falseNegatives=0, truePositives = 0, falsePositives = 0;	
	double IoUavg = 0;
	//Key points of ground truth rectangle
	cv::Point l2(groundTruth.x, groundTruth.y);
	cv::Point r2(groundTruth.x + groundTruth.width, groundTruth.y + groundTruth.height);
	double intersectArea = 0.0, falseArea = 0.0;

	for (int i = 0; i < boundRectData.size(); i++) {
		int count = 0;

		//Key points of bounding rectangles
		cv::Point l1(boundRectData[i].x, boundRectData[i].y);
		cv::Point r1(boundRectData[i].x + boundRectData[i].width, boundRectData[i].y + boundRectData[i].height);

		//Check for overlap resulting in false or true positive
		if (doOverlap(l1, r1, l2, r2)) {
			intersectArea += IntersectionArea(l1, r1, l2, r2);
			falseArea += nonIntersect(l1, r1, l2, r2);
			IoUavg += IntersectionOverUnion(l1, r1, l2, r2);
			truePositives++;
		}
		else {
			falseArea += boundRectData[i].width*boundRectData[i].height;
			falsePositives++;
		}
	}
	if ((truePositives == 0) && (groundTruth.x != 0) && (groundTruth.y != 0)) {
		falseNegatives++;
	}
	//Area evaluation 
	truePositiveArea.back().push_back(double(intersectArea / float(groundTruth.width*groundTruth.height)));
	falsePositiveArea.back().push_back(double(falseArea / float(intersectArea+falseArea)));
	precisionArea.back().push_back(double(intersectArea / float(intersectArea + falseArea)));
	
	//Count evaluation
	truePositiveCount.back().push_back(double(truePositives/ float(truePositives+falseNegatives)));
	falsePositiveCount.back().push_back(double(falsePositives / float(truePositives + falsePositives)));
	precisionCount.back().push_back(double(truePositives / float(truePositives + falsePositives)));

	//IoU
 	IoU.back().push_back(IoUavg);

}

void trueFalsePositiveRate(vector<vector<vector<vector<int>>>> boundRectData, vector<vector<int>> groundTruth, vector<vector<int>> &falsePositiveCount, vector<vector<int>> &truePositiveCount, 
	vector<vector<double>> &precision, vector<double> &recall,vector<vector<double>> &IoU, int GT_offset) {

	vector<vector<int>> falseNegativeCount;
	
	//Parameter
	for (int i = 0; i < boundRectData.size(); i++) {
		int count = 0, GTcount = 0;
		vector<double> IoUtime(0.0);
		vector<double> precisionTime(0.0);
		//vector<double> recallTime(0.0);
		falsePositiveCount.push_back(vector<int>(0));
		falseNegativeCount.push_back(vector<int>(0));
		truePositiveCount.push_back(vector<int>(0));
		int avgCount = 0;
		//Time
		for (int j = 0; j < boundRectData[i].size()-100; j++) {
			double IoUavg = 0.0;
			falsePositiveCount[i].push_back(0);
			falseNegativeCount[i].push_back(0);
			truePositiveCount[i].push_back(0);

			//Key points of ground truth rectangle
			cv::Point l2(groundTruth[GTcount][0], groundTruth[GTcount][1]);
			cv::Point r2(groundTruth[GTcount][0] + groundTruth[GTcount][2], groundTruth[GTcount][1] + groundTruth[GTcount][3]);

			if (count > GT_offset) {
				GTcount++;
			}
			count++;

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

double nonIntersect(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2) {

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
	double  nonIntersectArea = boxAArea + boxBArea - 2*interArea;
	return nonIntersectArea;
}

double IntersectionArea(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2) {

	int xA = max(l1.x, l2.x);
	int yA = max(l1.y, l2.y);
	int xB = min(r1.x, r2.x);
	int yB = min(r1.y, r2.y);

	//Compute the area of intersection rectangle
	double interArea = max(0, xB - xA) * max(0, yB - yA);

	return interArea;
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

	std::stringstream ss1;
	ss1 << File << "BoundRectSaliency.csv";
	boundRectFileSal = ss1.str();
	ss1.clear();

	std::stringstream ss2;
	ss2 << File << "avgTimeSaliency.csv";
	avgTimeFileSal = ss2.str();
	ss2.clear();

	std::stringstream ss3;
	ss3 << File << "BoundRectGMM.csv";
	boundRectFileGMM = ss3.str();
	ss3.clear();

	std::stringstream ss4;
	ss4 << File << "avgTimeGMM.csv";
	avgTimeFileGMM = ss4.str();
	ss4.clear();

	std::stringstream ss5;
	ss5 << File << "vectorlabel.csv";
	labelFile = ss5.str();
	ss5.clear();
}
//void writeEvaluationFile(vector<vector<vector<Rect>>> boundRectData, std::ofstream &File)