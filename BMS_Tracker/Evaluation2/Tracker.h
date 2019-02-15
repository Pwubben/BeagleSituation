#ifndef TRACKER_H
#define TRACKER_H
#define _USE_MATH_DEFINES
#include "BMS.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <memory>
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
	float heading;
};

struct EKFParams {
	double maxAcc, maxTurnRate, maxYawAcc, varGPS, varYaw, varYawRate;
};

struct matchedDetections {
	std::vector<double> relRange;
	std::vector<double> relAngle;
};

class Util {
public:
	static double deg2Rad(const double& degrees)
	{
		return ((degrees / 18000.0) * M_PI); //GPGLL Data needs division by 100 to obtain degrees
	}

	static double rad2Deg(const double& degrees)
	{
		return ((degrees / M_PI) * 180.0);
	}
};


class Kalman {
public:
	Kalman() {};
	Kalman(const Eigen::Vector2f& navdet, const float& vx, const float& vy);
	//Kalman(const float dt, const float& x, const float& y, const float& vx, const float& vy, const float& omega, const float& omegad);
	//void gainUpdate(const float& beta);
	void gainUpdate();
	void predict();
	void update(Eigen::Vector2f& selected_detections);
	Eigen::Vector2f getPrediction();

private:
	const double dt = 1 / 15; //15 FPS

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
	Eigen::Vector2f last_prediction;
	Eigen::Vector2f last_prediction_eigen;
	Eigen::Vector2f last_velocity;
	bool init;
};

class EKF {
public:
	EKF() {};
	//Beagle Constructor
	EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varYaw, double varYawRate, Eigen::VectorXf xInit);
	void compute(Eigen::Vector2f detection, double dt);
	void compute(Eigen::VectorXf detection);
	void updateQ(double dt);
	void updateJA(double dt);
	void predict();
	void update(const Eigen::VectorXf& Z, const Eigen::VectorXf& Hx, const Eigen::MatrixXf &JH, const Eigen::MatrixXf &R);
	struct beaglePrediction getBeaglePrediction(); //TODO getBeaglePrediction 

private:
	bool init;
	const int n = 5; // Number of states
	double dt; //Sample time

	double sGPS;
	double sCourse;
	double sVelocity;
	double sYaw;
	double sAccel;

	double _max_turn_rate;
	double _max_acceleration;
	double _max_yaw_accel;

	Eigen::MatrixXf P; // initial covaraince/uncertainity in states
	Eigen::MatrixXf Q; // process noise covariance
	Eigen::MatrixXf JH; // measurment jacobian
	Eigen::MatrixXf R; // measurement noise covariance
	Eigen::MatrixXf I; // Identity matrix
	Eigen::MatrixXf JA; // Jacobian state matrix
	Eigen::MatrixXf S; // Matrix for storing intermediate step in update part
	Eigen::MatrixXf K; // Kalman Gain
	Eigen::VectorXf Hx; // Measurement vector
	Eigen::VectorXf x; // State - x y heading velocity yaw_rat long_acceleration
};

class Track {
public:
	Track(double range, double angle, Eigen::Vector3f beagleMeas) {
		range_ = range;
		angle_ = angle;

		body2nav(range, angle, beagleMeas);

		//Target processing
		//Initial velocity estimates [m/s]

		//omega = 0;
		vxInit = 10.0;
		vyInit = 10.0;

		//Kalman filter track
		KF = std::make_unique<Kalman>(navDet, vxInit, vyInit);
	};

	void run(beaglePrediction _beaglePrediction);
	struct prediction getPrediction(); // based on protected values
	double getDetection();
	void setDetection(double range, double angle, Eigen::Vector3f beagleMeas);

	void body2nav(double range, double angle, Eigen::Vector3f& beagleMeas);
	void nav2body(beaglePrediction _beaglePrediction);

	int detectionAbsence = 0;

protected:
	std::unique_ptr<Kalman> KF;

	// Initialization
	float vxInit;
	float vyInit;

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

class DataAss {
public:
	DataAss::DataAss() : tracks_(){
		//Kalman filter Beagle
		Eigen::VectorXf xInit(5);
		xInit << 0.0, 0.0, 0.0, 14.0, 0.0; // Initiate velocity as it is not measured
		EKFParams params = { 1,400,1,0.05,0.05,0.05 };

		//Initiate EKF for Beagle
		BeagleTrack = std::make_unique<EKF>(params.maxAcc, params.maxTurnRate, params.maxYawAcc, params.varGPS, params.varYaw, params.varYawRate, xInit);
		beagleInit = true;
	}
	~DataAss() {
		
	};
	void run(struct detection info);

	void setBeagleData(Eigen::Vector4f& beagleData_);
	std::pair<bool, int > findInVector(const std::vector<double>& vecOfElements, const double& element);
	std::vector<double> distancePL(matchedDetections detection, prediction prdct);
	std::vector<double> distanceDet(std::vector<double> cdet, double rdet);

protected:
	std::unique_ptr<EKF> BeagleTrack;
	beaglePrediction _beaglePrediction;
	Eigen::Vector4f _beagleMeas;
	bool beagleInit;

	Eigen::Vector2f xyInit; //Initial position of Beagle
	const double earthRadius = 6378137; // Meters
	double aspectRatio;

	std::vector<Track> tracks_;
	double angleMatchThres = 5.0;
	double detectionMatchThres = 100;
	double absenceThreshold = 75;
	matchedDetections detect;

	Eigen::Vector2f beagleLocation;
	double beagleHeading;
};

class Detection {
public:
	Detection() {
		data_ass_ = std::make_unique<DataAss>();
	};
	
	void run(std::string File, std::string groundTruthFile, std::string beagleFile);
	void windowDetect(cv::Mat src,double max_dimension);
	void radarDetection(cv::Mat src);
	void saliencyDetection(cv::Mat src, double max_dimension, double sample_step, double threshold, std::vector<cv::Rect> GT);

	std::vector<std::vector<int>> readGroundTruth(std::string fileName);
	std::string getFileString(std::string fileName);
	std::vector<Eigen::Vector4f> loadBeagleData(std::string beagleFile);

protected:
	std::unique_ptr<DataAss> data_ass_;
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

	double radarRange = 926;
	double FOV = 60;

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


#endif
