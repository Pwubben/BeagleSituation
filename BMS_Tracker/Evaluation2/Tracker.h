#ifndef TRACKER_H
#define TRACKER_H
#define _USE_MATH_DEFINES
#include "BMS.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

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
	
protected:
	std::vector<Track> tracks_;
	double angleMatchThres = 5.0;
	double detectionMatchThres = 100;
};

class Track {
public:
	Track(double range, double angle) {
		range_ = range;
		angle_ = angle;
		body2nav();

		dt = 1 / 15; //15 FPS
		//Initial velocity estimates [m/s]
		vx_ = 10;
		vy_ = 10;
		omega = 0;
		Q << 10, 0,
			0, 5;
		R << 100, 0,
			0, 100;
		Kalman KF(dt, x_, y_, vx_, vy_);
	};

	void run();
	struct prediction getPrediction(); // based on protected values
	double getDetection();
	void setDetection(double range, double angle);
	cv::Point body2nav();

	int detectionAbsence = 0;
protected:
	//Last detections
	double range_;
	double angle_;

	double heading;

	//Navigation frame detections
	cv::Point navDet;

	//Prediction
	prediction prediction_;

	//States
	double dt;
	double x_;
	double y_;
	double vx_;
	double vy_;
	double omega_;
};

class Kalman {
public:
	Kalman(const float dt , const float& x, const float& y, const float& vx, const float& vy );
	
	void gainUpdate(const float& beta);
	void gainUpdate();
	cv::Point2f predict();
	Eigen::Vector4f update(Eigen::Vector2f& selected_detections);

private:
	Eigen::Matrix4f A; //Evolution state matrix
	Eigen::Matrix2f Q; //Covariance Matrix associated to the evolution process
	Eigen::MatrixXf G; //Evolution Noise Matrix
	Eigen::Matrix4f P; //Covariance Matrix
	Eigen::MatrixXf C;
	Eigen::Matrix2f R; //Proces measurement Covariance matrix
	Eigen::Matrix2f S;
	Eigen::MatrixXf K; //Gain
	Eigen::Matrix4f P_predict; //Covariance Matrix predicted error
	Eigen::Vector4f x_predict;
	Eigen::Vector4f x_filter;
	Eigen::Vector2f z_predict;
	cv::Point2f last_prediction;
	Eigen::Vector2f last_prediction_eigen;
	cv::Point2f last_velocity;
	bool init;
};

#endif
