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
	std::vector<double> radarVel;
	std::vector<double> cameraAngle;
	std::vector<double> cameraElevation;
};

struct evaluationSettings {
	bool cameraUtil;
	std::vector<int> dynamicModels;
	int objectChoice;
	double detectionThres;
	double maxDimension;
	double varianceFactor;
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
	std::vector<double> relVel;
};

struct Tuning {
	double rangeVarRadar;
	double angleVarRadar;

	double rangeVarCamera;
	double angleVarCamera;
	double varianceTimeFactor;

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

	static int factorial(int n) {
		return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
	}

	static bool diffVec(std::vector<int> tmp) {
		for (unsigned i = 0; i < tmp.size(); i++)
			for (unsigned k = i + 1; k < tmp.size(); k++)
				if (tmp[i] == tmp[k]) {
					return false;
				}
		return true;
	}

	static void makeCombiUtil(std::vector<std::vector<int>>& ans, std::vector<int>& tmp, int n, int left, int k)
	{
		// Pushing this vector to a vector of vector 
		if (k == 0) {
			if(Util::diffVec(tmp))
				ans.push_back(tmp);
			return;
		}

		// i iterates from left to n. First time 
		// left will be 0 
		for (int i = left; i <= n; ++i)
		{
			tmp.push_back(i);
			makeCombiUtil(ans, tmp, n, 0, k - 1);

			// Popping out last inserted element 
			// from the vector 
			tmp.pop_back();
		}
	}

	// Prints all combinations of size k of numbers 
	// from 0 to n. 
	static std::vector<std::vector<int>> makeCombi(int n, int k)
	{
		std::vector<std::vector<int>> ans;
		std::vector<int> tmp;
		Util::makeCombiUtil(ans, tmp, n, 0, k);
		return ans;
	}

	static std::string giveName(std::string video, double threshold, double imageSize) {
		std::stringstream ss;
		ss << video << "_" << threshold << "_" << imageSize << ".csv";
		std::string file = ss.str();
		return file;
	}

};

class KalmanFilters {
public:
	KalmanFilters() = default;
	virtual ~KalmanFilters() = default;
	virtual void compute(Eigen::VectorXd z, double radVel_ = 0, double angle_ = 0, Eigen::Vector3d beagleMeas = { 0,0,0 }) = 0;
	virtual Eigen::VectorXd getState() = 0;
	virtual Eigen::MatrixXd getCovariance() = 0;
	virtual double getProbability() = 0;

	virtual void setR(Eigen::MatrixXd R) = 0;
	virtual void setState(Eigen::VectorXd x_mixed) = 0;
	virtual void setCovariance(Eigen::MatrixXd P_) = 0;
	virtual void setMatchFlag(int mf) = 0;
};

class Kalman : public KalmanFilters {
public:
	Kalman() {};
	Kalman(const Eigen::Vector2d& navDet, const double& v, const double& heading, const Eigen::Matrix4d& Q_, const Eigen::Matrix4d& P_, const int& modelNumber = 0);
	
	//Kalman functions
	void compute(Eigen::VectorXd detection, double radVel_ = 0, double angle_ = 0, Eigen::Vector3d beagleMeas = { 0,0,0 }) override;
	void gainUpdate();
	void predict();
	void update(Eigen::VectorXd& selected_detections);

	//Get functions
	Eigen::MatrixXd getDynamicModel(int modelNumber);
	Eigen::Vector2d getPrediction();
	Eigen::VectorXd getState() override;
	Eigen::MatrixXd getCovariance() override;
	double getProbability() override;

	//Set functions
	void setMatchFlag(int mf) override;
	void setR(Eigen::MatrixXd R) override;
	void setState(Eigen::VectorXd x_mixed) override;
	void setCovariance(Eigen::MatrixXd P_) override;

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

	//IMM variables
	std::vector<double> modelTurnRates;
	int model;
	double lambda = 1;

	bool init;
};

class EKF : public KalmanFilters {
public:
	EKF() {};
	//Beagle Constructor
	EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varYaw, double varYawRate, Eigen::VectorXd xInit);
	EKF(const Eigen::Vector2d & navDet, const double& v, const double& heading, const Eigen::MatrixXd & Q_, const Eigen::MatrixXd & P_, const int & modelNumber = 5, Eigen::Vector3d beagleMeas = { 0,0,0 });

	bool BeagleObject = false;

	//Kalman functions
	void compute(Eigen::Vector2d detection, double dt);
	void compute(Eigen::VectorXd detection, double radVel_ = 0, double angle_ = 0, Eigen::Vector3d beagleMeas = { 0,0,0 }) override;
	void updateJA(double dt);
	void predict();
	void update(const Eigen::VectorXd& Z, double angle_ = 0, Eigen::Vector3d beagleMeas = { 0,0 ,0});

	double modelProbability(Eigen::MatrixXd P, Eigen::MatrixXd R, const Eigen::VectorXd & z);

	//Get Functions
	Eigen::Vector2d getPrediction();
	Eigen::Vector4d getBeaglePrediction(); 
	std::vector<std::vector<double>> getPlotVectors();
	Eigen::VectorXd getState() override;
	Eigen::MatrixXd getCovariance() override;
	double getProbability() override;

	//Set Functions
	void setState(Eigen::VectorXd x_mixed) override;
	void setCovariance(Eigen::MatrixXd P_) override;
	void setMatchFlag(int mf) override;
	void setR(Eigen::MatrixXd R) override;

private:
	bool init;
	const int n = 5; // Number of states
	double dt; //Sample time
	int modelNum;
	int matchFlag;

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
	
	Eigen::MatrixXd P; // initial covariance/uncertainity in states
	Eigen::MatrixXd Q; // process noise covariance
	Eigen::MatrixXd JH; // measurment jacobian
	Eigen::MatrixXd R; // measurement noise covariance
	Eigen::MatrixXd I; // Identity matrix
	Eigen::MatrixXd JA; // Jacobian state matrix
	Eigen::MatrixXd S; // Matrix for storing intermediate step in update part
	Eigen::MatrixXd K; // Kalman Gain
	Eigen::VectorXd Hx; // Measurement vector
	Eigen::VectorXd x; // State - x y heading velocity yaw_rat long_acceleration

	//IMM variables
	double lambda = 1;
};


class IMM {
public:
	IMM(const int& modelNum, const std::vector<int>& modelNumbers, const std::vector<Eigen::MatrixXd>& Q_, const std::vector<Eigen::MatrixXd>& P_, const Eigen::Vector2d& navDet, const double& vInit, const double& headingInit, Eigen::Vector3d beagleMeas = { 0,0,0 });

	void run(Eigen::VectorXd z, double radVel, double angle_, Eigen::VectorXd beagleMeas);
	void stateInteraction();
	void modelProbabilityUpdate();
	void stateEstimateCombination();

	//Set functions
	void setStateTransitionProbability(Eigen::MatrixXd stateTransitionProb);
	void setR(std::vector<Eigen::MatrixXd> Rvec_);
	void setMatchFlag(int mf);

	//Get functions
	Eigen::Vector2d getPrediction();
	Eigen::VectorXd getState();
	Eigen::VectorXd getMu();

private:
	std::vector<std::unique_ptr<KalmanFilters>> filters;
	std::vector<Eigen::MatrixXd> dynamicModels;

	bool init = true;
	int numStates = 5;
	const double dt = 1 / double(15);
	int matchFlag;
	
	//Kalman variables
	std::vector<Eigen::MatrixXd> Rvec;

	//State interaction variables
	Eigen::MatrixXd x_model;
	std::vector<Eigen::MatrixXd> P_model;
	Eigen::MatrixXd x_mixed;
	std::vector<Eigen::MatrixXd> P_mixed;
	Eigen::MatrixXd mu_tilde;
	Eigen::MatrixXd stateTransitionProb;

	//Model probability update
	Eigen::VectorXd lambda; 
	Eigen::VectorXd mu_hat;

	//State estimate combination
	Eigen::VectorXd x;
	Eigen::MatrixXd P;
};


class Track {
public:
	Track(double range, double angle, const double& velocity, Eigen::Vector4d beagleMeas, int objectChoice_, evaluationSettings settings);

	void run(Eigen::Vector3d _beaglePrediction);
	void tuning();
	void updateR(Eigen::Vector3d _beaglePrediction);
	struct prediction getPrediction(); // based on protected values
	double getDetection();
	double getAngle();
	void setDetection(const double& range, const double& angle, const double& velocity, Eigen::Vector4d beagleMeas, int matchFlag);

	void body2nav(const double& range, const double& angle, Eigen::Vector4d& beagleMeas);
	void nav2body(Eigen::Vector3d _beaglePrediction);

	int detectionAbsence;

	std::vector<std::vector<double>> getPlotVectors();
	std::vector<std::vector<Eigen::VectorXd>> getResultVectors();
protected:
	double count = 0;
	evaluationSettings evalSettings;
	std::unique_ptr<Kalman> KF;
	std::unique_ptr<EKF> EKF_;
	std::unique_ptr<IMM> IMM_;

	int objectChoice;
	int modelNum;
	std::vector<int> modelNumbers;

	// Initialization
	double vxInit;
	double vyInit;

	std::vector<Tuning> tuningVec;
	std::vector<Eigen::MatrixXd> Pvec;
	std::vector<Eigen::MatrixXd> Qvec;

	Eigen::MatrixXd g;
	double rangeVarRadar;
	double angleVarRadar;
	double rangeVarCamera;
	double angleVarCamera;
	double varianceTimeFactor;

	//Last detections
	double range_;
	double angle_;
	double radVel_;
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
	Eigen::VectorXd navEkf;

	//Prediction
	Eigen::Vector2d prediction_coord;
	prediction prediction_;

	//Results
	std::vector<Eigen::VectorXd> radarMeasurement;
	std::vector<Eigen::VectorXd> cameraMeasurement;
	std::vector<Eigen::VectorXd> stateVector;
	std::vector<Eigen::VectorXd> muVector;

};

class DataAss {
public:
	DataAss::DataAss(struct evaluationSettings settings) : tracks_(), absenceThreshold(300), objectChoice(settings.objectChoice), evalSettings(settings), cameraUtil(settings.cameraUtil)  {
		//Kalman filter Beagle
		Eigen::VectorXd xInit(5);
		xInit << 0.0, 0.0, 0.0, 8.0, 0.0; // Initiate velocity as it is not measured
		EKFParams params = { 0.1, 200.0, 0.1, 0.05, 0.05, 0.05 };

		//Initiate EKF for Beagle
		BeagleTrack = std::make_unique<EKF>(params.maxAcc, params.maxTurnRate, params.maxYawAcc, params.varGPS, params.varYaw, params.varYawRate, xInit);
		beagleInit = true;

		//True radial velocity state
		//Initiate EKF for Target
		TargetTrack = std::make_unique<EKF>(params.maxAcc, params.maxTurnRate, params.maxYawAcc, params.varGPS, params.varYaw, params.varYawRate, xInit);
	}
	~DataAss() {
		
	};
	void run(const struct detection& info);


	void setBeagleData(Eigen::Vector4d& beagleData_);
	void setTargetData(Eigen::Vector4d& targetData_);
	std::vector<std::vector<Eigen::VectorXd>> getStateVectors();
	std::vector<std::vector<std::vector<Eigen::VectorXd>>> getResultVectors();

	void NearestNeighbor(detection info);
	void GlobalNearestNeighbor(const detection& info, std::vector<prediction> predictionVector,std::vector<bool>& unassignedDetection, std::vector<double> lastDetection, std::vector<int> matchFlag);
	
	std::pair<bool, int> findInVector(const std::vector<double>& vecOfElements, const double& element);
	std::pair<bool, int> findRangeVector(const std::vector<double>& vecOfElements, const double& element, const double& range, const double& elementnr = -1);
	std::vector<double> distancePL(matchedDetections detection, prediction prdct);
	std::vector<int> distancePL(matchedDetections detection, std::vector<prediction> prdct);
	std::pair<std::vector<double>, std::vector<double>> distanceDet(std::vector<double> cdet, std::vector<double> hdet, double rdet);
	std::vector<int> DataAss::distanceDet(const detection& info, std::vector<prediction> rdet);
	void drawResults();

protected:
	int drawCount = 0;
	int radarCount = 0;
	std::unique_ptr<EKF> BeagleTrack;
	Eigen::Vector4d _beaglePrediction;
	Eigen::Vector4d _beagleMeas;
	Eigen::Vector4d _targetMeas;
	bool beagleInit;

	//Ground truth radial velocity measurement
	std::unique_ptr<EKF> TargetTrack;
	std::vector<Eigen::VectorXd> TargetState;
	std::vector<Eigen::VectorXd> BeagleState;

	//ResultVectors
	std::vector<std::vector<Eigen::VectorXd>> resultVector;
	struct evaluationSettings evalSettings;
	bool cameraUtil;

	Eigen::Vector2d xyInit; //Initial position of Beagle
	const double earthRadius = 6378137; // Meters
	double aspectRatio;

	std::vector<Track> tracks_;
	int objectChoice;
	double angleMatchThres = Util::deg2Rad(8);
	double detectionMatchThres = 10000;
	double absenceThreshold;
	matchedDetections detect;

	Eigen::Vector2d beagleLocation;
	double beagleHeading;
};

class Detection {
public:
	Detection(struct evaluationSettings settings) : evalSettings(settings), threshold(settings.detectionThres){
		data_ass_ = std::make_unique<DataAss>(settings);
	};
	~Detection() {}

	void run(std::string path, std::string File, std::string beagleFile, std::string radarFile, std::string targetFile, std::string beagleDes, std::string targetDes, std::string resultDes, int targets);
	void windowDetect(cv::Mat src, double max_dimension);
	void radarDetection(cv::Mat src);
	void saliencyDetection(cv::Mat src, double max_dimension, double sample_step, double threshold, cv::Rect GT);

	std::vector<std::vector<int>> readGroundTruth(std::string fileName);
	std::string getFileString(std::string fileName);
	std::vector<Eigen::Vector4d> loadBeagleData(std::string beagleFile);
	std::vector<Eigen::Vector4d> loadTargetData(std::string targetFile);
	std::vector<Eigen::VectorXd> loadRadarData(std::string radarFile,int targets);

	void writeDataFile(std::vector<std::vector<Eigen::VectorXd>> stateVectors, std::string BeagleFile, std::string TargetFile);
	void writeResultFile(std::vector<std::vector<std::vector<Eigen::VectorXd>>> stateVectors, std::string resultFile);

protected:
	bool centerInit = false;
	std::unique_ptr<DataAss> data_ass_;
	void getInput();

	struct detection info;
	evaluationSettings evalSettings;
	//Detection variables
	double threshold;
	int dilation_width_1 = 3;
	int dilation_width_2 = 3;
	double blur_std = 3;
	bool use_normalize = 1;
	bool handle_border = 0;
	int colorSpace = 1;
	bool whitening = 0;

	double radarRange = 1389;
	double FOV;

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
	std::string path_;
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
