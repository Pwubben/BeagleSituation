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
	std::vector<float> radarRange;
	std::vector<float> radarAngle;
	std::vector<float> cameraAngle;
};

struct beagleData {
	float heading;
	float turnRate;
	Eigen::Matrix2f location;
};

struct prediction {
	float range;
	float angle;
};

struct beaglePrediction {
	Eigen::Vector2f position;
	float heading;
};

struct EKFParams {
	float maxAcc, maxTurnRate, maxYawAcc, varGPS, varYaw, varYawRate;
};

struct matchedDetections {
	std::vector<float> relRange;
	std::vector<float> relAngle;
};

class Util {
public:
	static float deg2Rad(const float& degrees)
	{
		return ((degrees / float(180.0)) * M_PI); //GPGLL Data needs division by 100 to obtain degrees
	}

	static float rad2Deg(const float& degrees)
	{
		return ((degrees / M_PI) * 180.0);
	}

	static int sgn(float val) {
		return (0.0 < val) - (val < 0.0);
	}

	static float round(float var)
	{
		// 37.66666 * 100 =3766.66 
		// 3766.66 + .5 =37.6716    for rounding off value 
		// then type cast to int so value is 3766 
		// then divided by 100 so the value converted into 37.66 
		float value = (int)(var * 100 + .5);
		return (float)value / 100;
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
	void setMatchFlag(int mf);

	void setR(Eigen::MatrixXf R);

private:
	const float dt = 1 / float(15); //15 FPS

	int matchFlag;
	Eigen::Matrix4f A; //Evolution state matrix
	Eigen::Matrix4f Q; //Covariance Matrix associated to the evolution process
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
	EKF(float max_acceleration, float max_turn_rate, float max_yaw_accel, float varGPS, float varYaw, float varYawRate, Eigen::VectorXf xInit);
	void compute(Eigen::Vector2f detection, float dt);
	void compute(Eigen::VectorXf detection);
	void updateQ(float dt);
	void updateJA(float dt);
	void predict();
	void update(const Eigen::VectorXf& Z, const Eigen::VectorXf& Hx, const Eigen::MatrixXf &JH, const Eigen::MatrixXf &R);
	Eigen::Vector3f getBeaglePrediction(); //TODO getBeaglePrediction 
	std::vector<std::vector<float>> getPlotVectors();

private:
	bool init;
	const int n = 5; // Number of states
	float dt; //Sample time

	float sGPS;
	float sCourse;
	float sVelocity;
	float sYaw;
	float sAccel;

	float _max_turn_rate;
	float _max_acceleration;
	float _max_yaw_accel;

	//Plot vectors 
	std::vector<float> x_measVec;
	std::vector<float> y_measVec;
	std::vector<float> x_predictVec;
	std::vector<float> y_predictVec;
	
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
	Track(float range, float angle, Eigen::Vector3f beagleMeas); 

	void run(Eigen::Vector3f _beaglePrediction, int matchFlag);
	void updateR(Eigen::Vector3f _beaglePrediction);
	struct prediction getPrediction(); // based on protected values
	float getDetection();
	void setDetection(const float& range, const float& angle, Eigen::Vector3f beagleMeas);

	void body2nav(float range, float angle, Eigen::Vector3f& beagleMeas);
	void nav2body(Eigen::Vector3f _beaglePrediction);

	int detectionAbsence;

	std::vector<std::vector<float>> getPlotVectors();
protected:
	std::unique_ptr<Kalman> KF;

	// Initialization
	float vxInit;
	float vyInit;
	Eigen::MatrixXf R;
	Eigen::MatrixXf Rr;
	Eigen::MatrixXf Rc;
	Eigen::MatrixXf g;
	float rangeVar;
	float angleVar;
	float varianceTimeFactor;

	//Last detections
	float range_;
	float angle_;
	float heading;
	int matchFlag_;

	std::vector<float> x_measVec;
	std::vector<float> y_measVec;
	std::vector<float> x_predictVec;
	std::vector<float> y_predictVec;

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
	DataAss::DataAss() : tracks_(), absenceThreshold(300) {
		//Kalman filter Beagle
		Eigen::VectorXf xInit(5);
		xInit << 0.0, 0.0, 0.0, 14.0, 0.0; // Initiate velocity as it is not measured
		EKFParams params = { 0.1, 200.0, 0.1, 0.05, 0.05, 0.05 };

		//Initiate EKF for Beagle
		BeagleTrack = std::make_unique<EKF>(params.maxAcc, params.maxTurnRate, params.maxYawAcc, params.varGPS, params.varYaw, params.varYawRate, xInit);
		beagleInit = true;
	}
	~DataAss() {
		
	};
	void run(const struct detection& info);

	void setBeagleData(Eigen::Vector4f& beagleData_);
	std::pair<bool, int> findInVector(const std::vector<float>& vecOfElements, const float& element);
	std::pair<bool, int> findRangeVector(const std::vector<float>& vecOfElements, const float& element, const float& range);
	std::vector<float> distancePL(matchedDetections detection, prediction prdct);
	std::vector<float> distanceDet(std::vector<float> cdet, float rdet);

	void drawResults();

protected:
	int drawCount = 0;
	int radarCount = 0;
	std::unique_ptr<EKF> BeagleTrack;
	Eigen::Vector3f _beaglePrediction;
	Eigen::Vector4f _beagleMeas;
	bool beagleInit;

	Eigen::Vector2f xyInit; //Initial position of Beagle
	const float earthRadius = 6378137; // Meters
	float aspectRatio;

	std::vector<Track> tracks_;
	float angleMatchThres = 5.0;
	float detectionMatchThres = 10000;
	float absenceThreshold;
	matchedDetections detect;

	Eigen::Vector2f beagleLocation;
	float beagleHeading;
};

class Detection {
public:
	Detection() {
		data_ass_ = std::make_unique<DataAss>();
	};
	
	void run(std::string File, std::string groundTruthFile, std::string beagleFile);
	void windowDetect(cv::Mat src, float max_dimension);
	void radarDetection(cv::Mat src);
	void saliencyDetection(cv::Mat src, float max_dimension, float sample_step, float threshold, std::vector<cv::Rect> GT);

	std::vector<std::vector<int>> readGroundTruth(std::string fileName);
	std::string getFileString(std::string fileName);
	std::vector<Eigen::Vector4f> loadBeagleData(std::string beagleFile);

protected:
	bool centerInit = false;
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

	float radarRange = 926;
	float FOV = Util::deg2Rad(60);

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
	float thr;
	cv::Mat mask_trh;
	cv::Mat masked_img;
	cv::Mat sResult;
	cv::Mat mean;
	cv::Mat std;
	cv::Mat src_gray;
	cv::Mat src_small;
}; 


#endif
