#ifndef TRACKER_H
#define TRACKER_H

#include "BMS.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct detection {
	vector<double> radarRange;
	vector<double> radarAngle;
	vector<double> cameraAngle;
};

struct prediction {
	double range;
	double angle;
};

class Detection {
public:
	Detection() {	
		data_ass_();
	};
	
	void run(std::string File, std::string groundTruthFile, int GT_offset, int stopFrame);
	void windowDetect(cv::Mat src,double max_dimension);
	void radarDetection(cv::Mat src);
	void saliencyDetection(cv::Mat src, double max_dimension, double sample_step, double threshold, std::vector<cv::Rect> GT, int GT_offset, int stopFrame);
	std::vector<std::vector<int>> readGroundTruth(std::string fileName);
	std::string getFileString(std::string fileName);

	

protected:
	DataAss data_ass_;
	void getInput();

	struct detection info;

	//Detection variables
	int dilation_width_1 = 3;
	int dilation_width_2 = 3;
	float blur_std = 3;
	bool use_normalize = 1;
	bool handle_border = 0;
	int colorSpace = 1;
	bool whitening = 0;

	double radarRange = 1000;
	double FOV = 100;
	//Capture information
	cv::Rect seaWindow;
	cv::Rect radarWindow;
	int radarRadius; 
	cv::Point radarCenter;
	float width;
	float height;
	float maxD;
	cv::Size resizeDim;

	//Algorithm variables
	double thr;
	cv::Mat mask_trh;
	cv::Mat masked_img;
	cv::Mat sResult;
	cv::Mat mean;
	cv::Mat std;
	cv::Mat src_gray;
	cv::Mat src_small;
}; 

class DataAss {
public:
	D(int value) {
		tracks_();
	};

	void run(struct detection info);
	//{
	//	struct b blabla;
	//	for (Track tr : tracks_) {
	//		b = tr.GetPrediction();
	//		... // do something with b
	//	}
	//	//Compare detections with predictions
	//	//Assign nearest neighbor
	//	struct a assigned_detection = ...

	//	//If detections occur which cannot be matched
	//	//initiate new track
	//	Track tracki();
	//	tracks_.push_back(tracki);
	//	
	//	//If track has not been updated for some time
	//	//Terminate track
	//	track_[i].remove();
	//	
	//	//Run tracking algorithm		
	//	for (Track tr : tracks_) {
	//		tr.run(assigned_detection);
	//		tr.change_a(); // error
	//	}
	//}
protected:
	std::vector<Track> tracks_;
	double angleMatchThres = 5.0;
};

class Track {
public:
	Track() {
		a = 0;
		b = b_;
	};
	void run(a a_d) {
		//Do tracking stuff
	}
	struct prediction getPrediction(); // based on protected values
	double getDetection();
protected:
	//States
	int a;
	int b;
	void change_a();
};

#endif
