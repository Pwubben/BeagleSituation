#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>



void Track::run(beaglePrediction _beaglePrediction) {
	//Target processing
	//Track prediction
	KF->predict();
	//KF gain update
	KF->gainUpdate();
	//Track update
	KF->update(navDet);

	//Compute detection prediction by combining estimates from Beagle and Track
	nav2body(_beaglePrediction);
}

prediction Track::getPrediction() {
	return prediction_;
}

double Track::getDetection() {
	return range_;
}

void Track::setDetection(double range, double angle, Eigen::Vector3f beagleMeas) {
	if (detectionAbsence == 0) {
		range_ = range;
		angle_ = angle;
	}

	//Compute detection in navigation frame coordinates
	body2nav(range, angle, beagleMeas); //TODO Body2Nav - Perhaps we need predictions later instead of measurements
}

void Track::body2nav(double range, double angle, Eigen::Vector3f& beagleMeas) {
	relDet << sin(angle / 180.0 * M_PI)*range, cos(angle / 180.0 * M_PI)*range;

	rotMat << cos(beagleMeas(2)), -sin(beagleMeas(2)),
			  sin(beagleMeas(2)),  cos(beagleMeas(2));
	
	//Compute detection in navigation frame coordinates
	navDet = rotMat*relDet + beagleMeas.head(2);

}

void Track::nav2body(beaglePrediction _beaglePrediction) {

	//x y prediction target relative to beagle position prediction
	Eigen::Vector2f pdTarget = KF->getPrediction() - _beaglePrediction.position;

	//Rotate to body frame using beagle heading prediction
	rotMat << cos(_beaglePrediction.heading), sin(_beaglePrediction.heading),
		-sin(_beaglePrediction.heading), cos(_beaglePrediction.heading);

	//Compute detection in body frame coordinates
	prediction_coord = rotMat*pdTarget;

	//Compute range and angle
	prediction_.range = sqrt(pow(prediction_coord[0], 2) + pow(prediction_coord[1], 2));
	prediction_.angle = atan2(prediction_coord[0], prediction_coord[1])/M_PI*180.0;

	
}