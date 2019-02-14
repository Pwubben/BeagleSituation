#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

void Track::run() {
	//Target processing

	//KF gain update
	KF.gainUpdate();
	//Track update
	KF.update(navDet);
	//Track prediction
	KF.predict();

	//Compute detection prediction by combining estimates from Beagle and Track
	nav2body();
}

prediction Track::getPrediction() {
	return prediction_;
}

double Track::getDetection() {
	return range_;
}

void Track::setDetection(double range, double angle, double heading, Eigen::Vector2f beagleLocation) {
	range_ = range;

	//Compute detection in navigation frame coordinates
	body2nav(range, angle,heading, beagleLocation); //TODO Body2Nav - Perhaps we need predictions later instead of measurements
}

void Track::body2nav(double range, double angle, double heading, Eigen::Vector2f beagleLocation) {
	relDet << sin(angle / 180 * M_PI)*range, cos(angle / 180 * M_PI)*range;

	rotMat << cos(heading), -sin(heading),
			  sin(heading),  cos(heading);
	
	//Compute detection in navigation frame coordinates
	navDet = rotMat*relDet + beagleLocation;

}

Eigen::Vector2f Track::nav2body() {
	//Obtain prediction from Beagle KF
	beaglePrediction beaglePrediction_ = DataAss::getBeaglePrediction();

	//x y prediction target relative to beagle position prediction
	Eigen::Vector2f pdTarget = KF.getPrediction() - beaglePrediction_.position;

	//Rotate to body frame using beagle heading prediction
	rotMat << cos(beaglePrediction_.heading), sin(beaglePrediction_.heading),
		-sin(beaglePrediction_.heading), cos(beaglePrediction_.heading);

	//Compute detection in body frame coordinates
	prediction_coord = rotMat*pdTarget;

	//Compute range and angle
	prediction_.range = sqrt(pow(prediction_coord[0], 2) + pow(prediction_coord[1], 2));
	prediction_.angle = atan2(prediction_coord[0], prediction_coord[1])/M_PI*180;
}