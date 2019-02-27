#ifndef TRACKER_H
#define TRACKER_H
#define _USE_MATH_DEFINES
#include "BMS.h"
#include <iostream>
#include <algorithm>
//#include "GnuGraph.h"
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
	Eigen::Matrix2d location;
};

struct prediction {
	double range;
	double angle;
};

struct beaglePrediction {
	Eigen::Vector2d position;
	double heading;
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
		return ((degrees / double(180.0)) * M_PI); //GPGLL Data needs division by 100 to obtain degrees
	}

	static double rad2Deg(const double& degrees)
	{
		return ((degrees / M_PI) * 180.0);
	}

	static int sgn(double val) {
		return (0.0 < val) - (val < 0.0);
	}

	static double round(double var)
	{
		// 37.66666 * 100 =3766.66 
		// 3766.66 + .5 =37.6716    for rounding off value 
		// then type cast to int so value is 3766 
		// then divided by 100 so the value converted into 37.66 
		double value = (int)(var * 100 + .5);
		return (double)value / 100;
	}
};

class KalmanFilters {
public:
	KalmanFilters() {}
	virtual void compute(Eigen::VectorXd z) {}
};

class Kalman : public KalmanFilters {
public:
	Kalman() {};
	Kalman(const Eigen::Vector2d& navdet, const double& vx, const double& vy);
	//Kalman(const double dt, const double& x, const double& y, const double& vx, const double& vy, const double& omega, const double& omegad);
	//void gainUpdate(const double& beta);
	void gainUpdate();
	void predict();
	void update(Eigen::Vector2d& selected_detections);
	Eigen::Vector2d getPrediction();
	void setMatchFlag(int mf);

	void setR(Eigen::MatrixXd R);

private:
	const double dt = 1 / double(15); //15 FPS

	int matchFlag;
	Eigen::Matrix4d A; //Evolution state matrix
	Eigen::Matrix4d Q; //Covariance Matrix associated to the evolution process
	Eigen::MatrixXd G; //Evolution Noise Matrix
	Eigen::Matrix4d P; //Covariance Matrix
	Eigen::MatrixXd C;
	Eigen::Matrix2d R; //Process measurement Covariance matrix
	Eigen::Matrix2d S;
	Eigen::MatrixXd K; //Full measurement gain
	Eigen::Matrix4d P_predict; //Covariance Matrix predicted error
	Eigen::Vector4d x_predict;
	Eigen::Vector4d x_filter;
	Eigen::Vector2d z_predict;
	Eigen::Vector2d last_prediction;
	Eigen::Vector2d last_prediction_eigen;
	Eigen::Vector2d last_velocity;
	bool init;
};

class EKF : public KalmanFilters {
public:
	EKF() {};
	//Beagle Constructor
	EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varYaw, double varYawRate, Eigen::VectorXd xInit);
	void compute(Eigen::Vector2d detection, double dt);
	void compute(Eigen::VectorXd detection);
	void updateQ(double dt);
	void updateJA(double dt);
	void predict();
	void update(const Eigen::VectorXd& Z);
	Eigen::Vector3d getBeaglePrediction(); 
	std::vector<std::vector<double>> getPlotVectors();

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

	//Plot vectors 
	std::vector<double> x_measVec;
	std::vector<double> y_measVec;
	std::vector<double> x_predictVec;
	std::vector<double> y_predictVec;
	
	Eigen::MatrixXd P; // initial covaraince/uncertainity in states
	Eigen::MatrixXd Q; // process noise covariance
	Eigen::MatrixXd JH; // measurment jacobian
	Eigen::MatrixXd R; // measurement noise covariance
	Eigen::MatrixXd I; // Identity matrix
	Eigen::MatrixXd JA; // Jacobian state matrix
	Eigen::MatrixXd S; // Matrix for storing intermediate step in update part
	Eigen::MatrixXd K; // Kalman Gain
	Eigen::VectorXd Hx; // Measurement vector
	Eigen::VectorXd x; // State - x y heading velocity yaw_rat long_acceleration
};


class IMM {
public:
	IMM();
	~IMM();


private:
	std::vector<std::unique_ptr<KalmanFilters>> filters;
};


class Track {
public:
	Track(double range, double angle, Eigen::Vector3d beagleMeas); 

	void run(Eigen::Vector3d _beaglePrediction);
	void updateR(Eigen::Vector3d _beaglePrediction);
	struct prediction getPrediction(); // based on protected values
	double getDetection();
	void setDetection(const double& range, const double& angle, Eigen::Vector3d beagleMeas, int matchFlag);

	void body2nav(const double& range, const double& angle, Eigen::Vector3d& beagleMeas);
	void nav2body(Eigen::Vector3d _beaglePrediction);

	int detectionAbsence;

	std::vector<std::vector<double>> getPlotVectors();
protected:
	std::unique_ptr<Kalman> KF;

	// Initialization
	double vxInit;
	double vyInit;
	Eigen::MatrixXd R;
	Eigen::MatrixXd Rr;
	Eigen::MatrixXd Rc;
	Eigen::MatrixXd g;
	double rangeVarRadar;
	double angleVarRadar;
	double rangeVarCamera;
	double angleVarCamera;
	double varianceTimeFactor;

	//Last detections
	double range_;
	double angle_;
	double heading;
	int matchFlag_;

	std::vector<double> x_measVec;
	std::vector<double> y_measVec;
	std::vector<double> x_predictVec;
	std::vector<double> y_predictVec;

	//Navigation frame detections
	Eigen::Matrix2d rotMat;
	Eigen::Vector2d relDet;
	Eigen::Vector2d navDet;

	//Prediction
	Eigen::Vector2d prediction_coord;
	prediction prediction_;

};

class DataAss {
public:
	DataAss::DataAss() : tracks_(), absenceThreshold(300) {
		//Kalman filter Beagle
		Eigen::VectorXd xInit(5);
		xInit << 0.0, 0.0, 0.0, 14.0, 0.0; // Initiate velocity as it is not measured
		EKFParams params = { 0.1, 200.0, 0.1, 0.05, 0.05, 0.05 };

		//Initiate EKF for Beagle
		BeagleTrack = std::make_unique<EKF>(params.maxAcc, params.maxTurnRate, params.maxYawAcc, params.varGPS, params.varYaw, params.varYawRate, xInit);
		beagleInit = true;
	}
	~DataAss() {
		
	};
	void run(const struct detection& info);

	void setBeagleData(Eigen::Vector4d& beagleData_);
	std::pair<bool, int> findInVector(const std::vector<double>& vecOfElements, const double& element);
	std::pair<bool, int> findRangeVector(const std::vector<double>& vecOfElements, const double& element, const double& range);
	std::vector<double> distancePL(matchedDetections detection, prediction prdct);
	std::vector<double> distanceDet(std::vector<double> cdet, double rdet);

	void drawResults();

protected:
	int drawCount = 0;
	int radarCount = 0;
	std::unique_ptr<EKF> BeagleTrack;
	Eigen::Vector3d _beaglePrediction;
	Eigen::Vector4d _beagleMeas;
	bool beagleInit;

	Eigen::Vector2d xyInit; //Initial position of Beagle
	const double earthRadius = 6378137; // Meters
	double aspectRatio;

	std::vector<Track> tracks_;
	double angleMatchThres = Util::deg2Rad(8);
	double detectionMatchThres = 10000;
	double absenceThreshold;
	matchedDetections detect;

	Eigen::Vector2d beagleLocation;
	double beagleHeading;
};

class Detection {
public:
	Detection() {
		data_ass_ = std::make_unique<DataAss>();
	};
	
	void run(std::string File, std::string groundTruthFile, std::string beagleFile, std::string radarFile);
	void windowDetect(cv::Mat src, double max_dimension);
	void radarDetection(cv::Mat src);
	void saliencyDetection(cv::Mat src, double max_dimension, double sample_step, double threshold, std::vector<cv::Rect> GT);

	std::vector<std::vector<int>> readGroundTruth(std::string fileName);
	std::string getFileString(std::string fileName);
	std::vector<Eigen::Vector4d> loadBeagleData(std::string beagleFile);
	std::vector<Eigen::Vector2d> loadRadarData(std::string radarFile);

protected:
	bool centerInit = false;
	std::unique_ptr<DataAss> data_ass_;
	void getInput();

	struct detection info;

	//Detection variables
	int dilation_width_1 = 3;
	int dilation_width_2 = 3;
	double blur_std = 3;
	bool use_normalize = 1;
	bool handle_border = 0;
	int colorSpace = 1;
	bool whitening = 0;

	double radarRange = 1389;
	double FOV = Util::deg2Rad(120);

	//Capture information
	cv::Rect seaWindow;
	cv::Rect radarWindow;
	int radarRadius; 
	cv::Point radarCenter;
	double width;
	double height;
	double maxD;
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
