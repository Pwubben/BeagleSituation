#ifndef TRACKER_H
#define TRACKER_H
#define _USE_MATH_DEFINES
#include "BMS.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <math.h>

struct detection {
	std::vector<double> radarRange;
	std::vector<double> radarAngle;
	std::vector<double> cameraAngle;
};

struct beagleData {
	double heading;
	double turnRate;
	Eigen::Matrix2f location;
};

struct prediction {
	double range;
	double angle;
};

struct beaglePrediction {
	Eigen::Vector2f position;
	double heading;
};


class Detection {
public:
	Detection() {	
		DataAss data_ass_();
	};
	
	void run(std::string File, std::string groundTruthFile, std::string beagleFile, int GT_offset, int stopFrame);
	void windowDetect(cv::Mat src,double max_dimension);
	void radarDetection(cv::Mat src);
	void saliencyDetection(cv::Mat src, double max_dimension, double sample_step, double threshold, std::vector<cv::Rect> GT, int GT_offset, int stopFrame);
	std::vector<beagleData> loadBeagleData(std::string beagleFile);
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
	DataAss() {
		std::vector<Track> tracks_();
		//Kalman filter Beagle
		Kalman BeagleTrack();
	};

	void run(struct detection info);
	
	void setBeagleData(beagleData beagleData_); //TODO setBeagleData - Write when files are known
	static struct beaglePrediction getBeaglePrediction(); //TODO getBeaglePrediction 
	
protected:
	static Kalman BeagleTrack; //TODO BeagleTrack add class with Beagle Tracker
	std::vector<Track> tracks_;
	double angleMatchThres = 5.0;
	double detectionMatchThres = 100;

	Eigen::Vector2f beagleLocation;
	double beagleHeading;
};

class Track {
public:
	Track() {};
	Track(double range, double angle) {
		range_ = range;
		angle_ = angle;

		body2nav(range,angle);

		//Target processing
		dt = 1 / 15; //15 FPS

		//Initial velocity estimates [m/s]
		vx_ = 10;
		vy_ = 10;
		//omega = 0;
		
		//Kalman filter track
		Kalman KF(dt, x_, y_, vx_, vy_);

		
	};

	void run();
	struct prediction getPrediction(); // based on protected values
	double getDetection();
	void setDetection(double range, double angle, double heading, Eigen::Vector2f beagleLocation);

	//TODO
	void setBeagleData(beagleData beagleData_); //Write when files are known
	void body2nav(double range, double angle, double heading, Eigen::Vector2f beagleLocation);
	Eigen::Vector2f nav2body();

	int detectionAbsence = 0;

	
protected:
	DataAss data_ass_;
	Kalman KF;

	//Last detections
	double range_;
	double angle_;

	double heading;

	//Navigation frame detections
	Eigen::Matrix2f rotMat;
	Eigen::Vector2f relDet;
	Eigen::Vector2f navDet;

	//Prediction
	Eigen::Vector2f prediction_coord;
	prediction prediction_;

};

class Kalman {
public:
	Kalman() {};
	Kalman(const float dt , const float& x, const float& y, const float& vx, const float& vy );
	Kalman(const float dt, const float& x, const float& y, const float& vx, const float& vy, const float& omega, const float& omegad);
	//void gainUpdate(const float& beta);
	void gainUpdate();
	void predict();
	void update(Eigen::Vector2f& selected_detections);
	Eigen::Vector2f getPrediction();

private:
	Eigen::Matrix4f A; //Evolution state matrix
	Eigen::Matrix2f Q; //Covariance Matrix associated to the evolution process
	Eigen::MatrixXf G; //Evolution Noise Matrix
	Eigen::Matrix4f P; //Covariance Matrix
	Eigen::MatrixXf C;
	Eigen::Matrix2f R; //Process measurement Covariance matrix
	Eigen::Matrix2f S;
	Eigen::MatrixXf K; //Full measurement gain
	Eigen::Matrix4f P_predict; //Covariance Matrix predicted error
	Eigen::Vector4f x_predict;
	Eigen::Vector4f x_filter;
	Eigen::Vector2f z_predict;
	cv::Point2f last_prediction;
	Eigen::Vector2f last_prediction_eigen;
	cv::Point2f last_velocity;
	bool init;
};

class EKF {
public:
	EKF() {};
	EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varSpeed, double varYaw);
	void compute(Eigen::Vector2f detection,double dt);
	void updateQ(double dt);
	void updateJA(double dt);
	void predict();
	void update(const Eigen::VectorXd& Z, const Eigen::VectorXd& Hx, const Eigen::MatrixXd &JH, const Eigen::MatrixXd &R);
	
private:
	const int n = 5; // Number of states
	double dt;

	double sGPS;
	double sCourse;
	double sVelocity;
	double sYaw;
	double sAccel;

	double _max_turn_rate;
	double _max_acceleration;
	double _max_yaw_accel;

	Eigen::MatrixXd P; // initial covaraince/uncertainity in states
	Eigen::MatrixXd Q; // process noise covariance
	Eigen::MatrixXd JH; // measurment jacobian
	Eigen::MatrixXd R; // measurement noise covariance
	Eigen::MatrixXd I; // Identity matrix
	Eigen::MatrixXd JA; // Jacobian state matrix
	Eigen::MatrixXd S; // Matrix for storing intermediate step in update part
	Eigen::MatrixXd K; // Kalman Gain
	Eigen::VectorXd x; // State - x y heading velocity yaw_rat long_acceleration
};
#endif
