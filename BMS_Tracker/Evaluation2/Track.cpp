#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Track::Track(double range, double angle, Eigen::Vector3d beagleMeas, int objectChoice_) : 
	detectionAbsence(0),
	objectChoice(objectChoice_)
{
	range_ = range;
	angle_ = angle;

	body2nav(range, angle, beagleMeas);

	x_measVec.push_back(navDet(0));
	y_measVec.push_back(navDet(1));

	//IMM parameters
	modelNum = 4;
	modelNumbers = { 0, 1, 2, 5 };

	//Set initial matrices
	tuning();

	//Initial velocity estimates [m/s]
	double vInit = 6;
	double headingInit = atan2(beagleMeas(0) - navDet(0), beagleMeas(1) - navDet(1));

	//Kalman filter track
	if (objectChoice == 0)
		KF = std::make_unique<Kalman>(navDet, vInit, headingInit, Qvec[0], Pvec[0]);
	else if (objectChoice == 1)
		EKF_ = std::make_unique<EKF>(navDet, vInit, headingInit, Qvec[5], Pvec[5]);
	else
		IMM_ = std::make_unique<IMM>(modelNum, modelNumbers, Qvec, Pvec, navDet, vInit, headingInit);
}

void Track::tuning() {
	
	//Model 0 - Constant velocity
	tuningVec.push_back(Tuning({
		{0.05, 0,
		 0, 0.05},//Radar measurement noise covariance
		0.1,	  //rangeVarCamera
		0.1,      //angleVarCamera
		1.1       //varianceTimeFactor	
	}));
	Pvec.push_back(Eigen::MatrixXd(4, 4));
	Qvec.push_back(Eigen::MatrixXd(4, 4));
	Pvec.back() << 15, 0, 0, 0,
					0, 2, 0, 0,
					0, 0, 15, 0,
					0, 0, 0, 2;

	Qvec.back() << 20, 0, 0, 0,
					0, 3, 0, 0,
					0, 0, 20, 0,
					0, 0, 0, 3;

	//Model 1 & 2 - Constant turn 30deg/s
	tuningVec.insert(tuningVec.end(), 2, Tuning({
		{ 0.05, 0,
		0, 0.05 }, //Radar measurement noise covariance
		0.1,	   //rangeVarCamera
		0.1,       //angleVarCamera
		1.1        //varianceTimeFactor
	}));
	Pvec.push_back(Eigen::MatrixXd(4, 4));
	Qvec.push_back(Eigen::MatrixXd(4, 4));
	Pvec.back() << 15, 0, 0, 0,
					0, 2, 0, 0,
					0, 0, 15, 0,
					0, 0, 0, 2;

	Qvec.back() << 20, 0, 0, 0,
					0, 3, 0, 0,
					0, 0, 20, 0,
					0, 0, 0, 3;
	Pvec[1] = Pvec.back();
	Qvec[1] = Qvec.back();

	//Model 3 & 4 - Constant turn 50deg/s
	tuningVec.insert(tuningVec.end(), 2, Tuning({
		{ 0.05, 0,
		0, 0.05 }, //Radar measurement noise covariance
		0.1,	   //rangeVarCamera
		0.1,       //angleVarCamera
		1.1        //varianceTimeFactor
	}));
	Pvec.push_back(Eigen::MatrixXd(4, 4));
	Qvec.push_back(Eigen::MatrixXd(4, 4));
	Pvec.back() << 15, 0, 0, 0,
					0, 2, 0, 0,
					0, 0, 15, 0,
					0, 0, 0, 2;

	Qvec.back() << 20, 0, 0, 0,
					0, 3, 0, 0,
					0, 0, 20, 0,
					0, 0, 0, 3;
	Pvec[3] = Pvec.back();
	Qvec[3] = Qvec.back();

	//Model 5 - EKF - Constant Velocity Constant Turn
	tuningVec.push_back(Tuning({
		{ 0.05, 0,
		0, 0.05 }, //Radar measurement noise covariance
		0.1,	   //rangeVarCamera
		0.1,       //angleVarCamera
		1.1        //varianceTimeFactor
	}));
	Pvec.push_back(Eigen::MatrixXd(5, 5));
	Qvec.push_back(Eigen::MatrixXd(5, 5));
	Pvec.back() <<
		15, 0, 0, 0, 0,
		0, 15, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 20, 0,
		0, 0, 0, 0, 10;
	Qvec.back() <<
		20, 0, 0, 0, 0,
		0, 20, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 10, 0,
		0, 0, 0, 0, 20;
	}

void Track::run(Eigen::Vector3d _beaglePrediction) {
	
	
	updateR(_beaglePrediction);

	if (objectChoice == 0) {
		KF->setMatchFlag(matchFlag_);
		KF->compute(navDet);
	}
	else if (objectChoice == 1) {
		EKF_->setMatchFlag(matchFlag_);
		EKF_->compute(navDet);
	}
	else {
		IMM_->setMatchFlag(matchFlag_);
		IMM_->run(navDet);
	}
	
	//Compute detection prediction by combining estimates from Beagle and Track
	nav2body(_beaglePrediction);
}

void Track::updateR(Eigen::Vector3d _beaglePrediction) {
	
	g << sin(_beaglePrediction(2) + angle_), range_*cos(_beaglePrediction(2) + angle_),
	cos(_beaglePrediction(2) + angle_), -range_*sin(_beaglePrediction(2) + angle_);

	if (objectChoice == 0) {
		Eigen::Matrix2d Rc, R;
		if (matchFlag_ == 0)
			R = g*tuningVec[0].Rr*g.transpose();
		else if (matchFlag_ == 1) {
			Rc << tuningVec[0].rangeVarCamera *pow(tuningVec[0].varianceTimeFactor, detectionAbsence), 0,
				0, tuningVec[0].angleVarCamera;
			R = g*Rc*g.transpose();
		}
		KF->setR(R);
	}
	else if (objectChoice == 1) {
		Eigen::Matrix2d Rc, R;
		if (matchFlag_ == 0)
			R = g*tuningVec[5].Rr*g.transpose();
		else if (matchFlag_ == 1) {
			Rc << tuningVec[5].rangeVarCamera *pow(tuningVec[5].varianceTimeFactor, detectionAbsence), 0,
				0, tuningVec[5].angleVarCamera;
			R = g*Rc*g.transpose();
		}
		EKF_->setR(R);
	}
	else {
		std::vector<Eigen::MatrixXd> Rvec;
		for (auto &&i : modelNumbers) {
			Eigen::Matrix2d Rc, R;
			if (matchFlag_ == 0)
				R = g*tuningVec[i].Rr*g.transpose();
			else if (matchFlag_ == 1) {
				Rc << tuningVec[i].rangeVarCamera *pow(tuningVec[i].varianceTimeFactor, detectionAbsence), 0,
					0, tuningVec[i].angleVarCamera;
				R = g*Rc*g.transpose();
			}
			Rvec.push_back(R);
		}
		IMM_->setR(Rvec);
	}
}

prediction Track::getPrediction() {
	return prediction_;
}

double Track::getDetection() {
	return range_;
}

void Track::setDetection(const double& range,const double& angle, Eigen::Vector3d beagleMeas, int matchFlag) {
	matchFlag_ = matchFlag;

	if (detectionAbsence == 0) {
		std::cout << "\n" << std::endl;
		range_ = range;
		angle_ = angle;

		

	}

	//Compute detection in navigation frame coordinates
	body2nav(range, angle, beagleMeas); //TODO Body2Nav - Perhaps we need predictions later instead of measurements

	if (matchFlag_ != 2) {

		std::cout << "MatchFlag: "<< matchFlag_ <<  "- Range: " << range << "- Angle: " << Util::rad2Deg(angle) << std::endl;
		std::cout << "BeagleMeas " << beagleMeas(0) << " - " << beagleMeas(1) << " - " << beagleMeas(2) << std::endl;
		x_measVec.push_back(navDet(0));
		y_measVec.push_back(navDet(1));

		//std::cout << "x: " << navDet(0) << " - y: " << navDet(1) << "\n" << std::endl;
	}
}

void Track::body2nav(const double& range, const double& angle, Eigen::Vector3d& beagleMeas) {
	
	relDet << sin(angle+ beagleMeas(2))*range, cos(angle + beagleMeas(2))*range;

	//Compute detection in navigation frame coordinates
	navDet = relDet + beagleMeas.head(2);
}

void Track::nav2body(Eigen::Vector3d _beaglePrediction) {

	Eigen::Vector2d z_predict = KF->getPrediction();
	//x y prediction target relative to beagle position prediction
	Eigen::Vector2d pdTarget = z_predict - _beaglePrediction.head(2);

	//Create plot vectors containing predictions
	x_predictVec.push_back(z_predict(0));
	y_predictVec.push_back(z_predict(1));

	//Rotate to body frame using beagle heading prediction
	rotMat << cos(_beaglePrediction(2)), -sin(_beaglePrediction(2)),
		sin(_beaglePrediction(2)), cos(_beaglePrediction(2));

	//Compute detection in body frame coordinates
	prediction_coord = rotMat*pdTarget;

	//Compute range and angle
	prediction_.range = sqrt(pow(prediction_coord[0], 2) + pow(prediction_coord[1], 2));
	prediction_.angle = atan2(prediction_coord[0], prediction_coord[1]);

}

std::vector<std::vector<double>> Track::getPlotVectors() {
	std::vector<std::vector<double>> plotVectors;
	plotVectors.push_back(x_measVec);
	plotVectors.push_back(y_measVec);
	plotVectors.push_back(x_predictVec);
	plotVectors.push_back(y_predictVec);
	return plotVectors;
}

