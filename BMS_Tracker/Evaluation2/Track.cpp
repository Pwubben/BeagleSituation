#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Track::Track(double range, double angle, Eigen::Vector3d beagleMeas) : detectionAbsence(0) {
	range_ = range;
	angle_ = angle;

	body2nav(range, angle, beagleMeas);

	x_measVec.push_back(navDet(0));
	y_measVec.push_back(navDet(1));

	//Todo - varianties tunen
	rangeVarRadar = 0.05; 
	angleVarRadar = 0.05;

	rangeVarCamera = 0.1;
	angleVarCamera = 0.1;
	varianceTimeFactor = 1.1;

	//Radar measurement noise covariance matrix 
	Rr = Eigen::MatrixXd(2, 2);
	Rc = Eigen::MatrixXd(2, 2);
	R  = Eigen::MatrixXd(2, 2);
	//Transformation matrix
	g = Eigen::MatrixXd(2, 2);

	Rr << rangeVarRadar, 0,
		0, angleVarRadar;

	R = Rr;
	//Target processing
	//Initial velocity estimates [m/s]

	//omega = 0;

	double vInit = 6;
	double headingInit = atan2(beagleMeas(0) - navDet(0), beagleMeas(1) - navDet(1));
	vxInit = sin(headingInit)*vInit;
	vyInit = cos(headingInit)*vInit;

	//Kalman filter track
	KF = std::make_unique<Kalman>(navDet, vxInit, vyInit);
}

void Track::run(Eigen::Vector3d _beaglePrediction) {
	
	
	updateR(_beaglePrediction);

	KF->setMatchFlag(matchFlag_);

	//Target processing
	//Track prediction
	if (matchFlag_ < 2) {
		
		//KF gain update
		KF->gainUpdate();
		//Track update
		KF->update(navDet);

		KF->predict();
	}
	else {
		
		KF->update(navDet);

		KF->predict();
	}
	//Compute detection prediction by combining estimates from Beagle and Track
	nav2body(_beaglePrediction);
}

void Track::updateR(Eigen::Vector3d _beaglePrediction) {
	g << sin(_beaglePrediction(2) + angle_), range_*cos(_beaglePrediction(2) + angle_),
	cos(_beaglePrediction(2) + angle_), -range_*sin(_beaglePrediction(2) + angle_);

	if (matchFlag_ == 0)
		R = g*Rr*g.transpose();
	else if (matchFlag_ == 1) {
		Rc << rangeVarCamera*pow(varianceTimeFactor,detectionAbsence), 0, 
			0, angleVarCamera;
		R = g*Rc*g.transpose();
	}

	KF->setR(R);

	//std::cout << "g: " << g << std::endl;
	//std::cout << "R: " << R << std::endl;
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

