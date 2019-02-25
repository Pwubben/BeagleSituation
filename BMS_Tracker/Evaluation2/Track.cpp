#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Track::Track(float range, float angle, Eigen::Vector3f beagleMeas) : detectionAbsence(0) {
	range_ = range;
	angle_ = angle;

	body2nav(range, angle, beagleMeas);

	x_measVec.push_back(std::min(navDet(0), float(3000.0)));
	y_measVec.push_back(std::min(navDet(1), float(3000.0)));

	//Todo - varianties tunen
	rangeVar = 0.5; 
	angleVar = 0.5;
	varianceTimeFactor = 1.1;

	//Radar measurement noise covariance matrix 
	Rr = Eigen::MatrixXf(2, 2);
	Rc = Eigen::MatrixXf(2, 2);
	R  = Eigen::MatrixXf(2, 2);
	//Transformation matrix
	g = Eigen::MatrixXf(2, 2);

	Rr << rangeVar, 0,
		0, angleVar;

	R = Rr;
	//Target processing
	//Initial velocity estimates [m/s]

	//omega = 0;

	float vInit = 6;
	float headingInit = atan2(beagleMeas(0) - navDet(0), beagleMeas(1) - navDet(1));
	vxInit = sin(headingInit)*vInit;
	vyInit = cos(headingInit)*vInit;

	//Kalman filter track
	KF = std::make_unique<Kalman>(navDet, vxInit, vyInit);
}

void Track::run(Eigen::Vector3f _beaglePrediction, int matchFlag) {
	
	
	matchFlag_= matchFlag;

	updateR(_beaglePrediction);

	KF->setMatchFlag(matchFlag);

	//Target processing
	//Track prediction
	if (matchFlag < 2) {
		
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

void Track::updateR(Eigen::Vector3f _beaglePrediction) {
	g << sin(_beaglePrediction(2) + angle_), range_*cos(_beaglePrediction(2) + angle_),
	cos(_beaglePrediction(2) + angle_), -range_*sin(_beaglePrediction(2) + angle_);

	if (matchFlag_ == 0)
		R = g*Rr*g.transpose();
	else if (matchFlag_ == 1) {
		Rc << rangeVar*pow(detectionAbsence, varianceTimeFactor);
		R = g*Rc*g.transpose();
	}

	KF->setR(R);

	//std::cout << "g: " << g << std::endl;
	//std::cout << "R: " << R << std::endl;
}

prediction Track::getPrediction() {
	return prediction_;
}

float Track::getDetection() {
	return range_;
}

void Track::setDetection(const float& range,const float& angle, Eigen::Vector3f beagleMeas) {
	if (detectionAbsence == 0) {
		std::cout << "\n \n" << std::endl;
		range_ = range;
		angle_ = angle;
	}

	//Compute detection in navigation frame coordinates
	body2nav(range, angle, beagleMeas); //TODO Body2Nav - Perhaps we need predictions later instead of measurements

	if (matchFlag_ != 2) {
		x_measVec.push_back(std::min(navDet(0), float(3000.0)));
		y_measVec.push_back(std::min(navDet(1), float(3000.0)));

		//std::cout << "x: " << navDet(0) << " - y: " << navDet(1) << "\n" << std::endl;
	}
}

void Track::body2nav(float range, float angle, Eigen::Vector3f& beagleMeas) {
	

	relDet << sin(angle_)*range, cos(angle_)*range;

	rotMat << cos(beagleMeas(2)), sin(beagleMeas(2)),
			  -sin(beagleMeas(2)),  cos(beagleMeas(2));
	
	

	//Compute detection in navigation frame coordinates
	navDet = rotMat*relDet + beagleMeas.head(2);

}

void Track::nav2body(Eigen::Vector3f _beaglePrediction) {

	Eigen::Vector2f z_predict = KF->getPrediction();
	//x y prediction target relative to beagle position prediction
	Eigen::Vector2f pdTarget = z_predict - _beaglePrediction.head(2);

	//Create plot vectors containing predictions
	x_predictVec.push_back(std::min(z_predict(0), float(3000.0)));
	y_predictVec.push_back(std::min(z_predict(1), float(3000.0)));

	//Rotate to body frame using beagle heading prediction
	rotMat << cos(_beaglePrediction(2)), -sin(_beaglePrediction(2)),
		sin(_beaglePrediction(2)), cos(_beaglePrediction(2));

	//Compute detection in body frame coordinates
	prediction_coord = rotMat*pdTarget;

	//Compute range and angle
	prediction_.range = sqrt(pow(prediction_coord[0], 2) + pow(prediction_coord[1], 2));
	prediction_.angle = atan2(prediction_coord[0], prediction_coord[1]);

}

std::vector<std::vector<float>> Track::getPlotVectors() {
	std::vector<std::vector<float>> plotVectors;
	plotVectors.push_back(x_measVec);
	plotVectors.push_back(y_measVec);
	plotVectors.push_back(x_predictVec);
	plotVectors.push_back(y_predictVec);
	return plotVectors;
}

