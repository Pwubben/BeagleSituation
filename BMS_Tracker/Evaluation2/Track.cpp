#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

void Track::run(Eigen::Vector3f _beaglePrediction, int matchFlag) {
	matchFlag_= matchFlag;

	//KF->setMatchFlag(matchFlag);
	//Target processing
	//Track prediction
	if (matchFlag == 0) {
		KF->predict();
		//KF gain update
		KF->gainUpdate();
		//Track update
		KF->update(navDet);
	}
	if (matchFlag == 2) {
		KF->predict();
		KF->update(navDet);
	}
	//Compute detection prediction by combining estimates from Beagle and Track
	nav2body(_beaglePrediction);
}

prediction Track::getPrediction() {
	return prediction_;
}

float Track::getDetection() {
	return range_;
}

void Track::setDetection(const float& range,const float& angle, Eigen::Vector3f beagleMeas) {
	if (detectionAbsence == 0) {
		range_ = range;
		angle_ = angle;
	}

	//Compute detection in navigation frame coordinates
	body2nav(range, angle, beagleMeas); //TODO Body2Nav - Perhaps we need predictions later instead of measurements

	x_measVec.push_back(std::min(navDet(0), float(3000.0)));
	y_measVec.push_back(std::min(navDet(1), float(3000.0)));

}

void Track::body2nav(float range, float angle, Eigen::Vector3f& beagleMeas) {
	relDet << sin(angle / 180.0 * M_PI)*range, cos(angle / 180.0 * M_PI)*range;

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
	prediction_.angle = atan2(prediction_coord[0], prediction_coord[1])/M_PI*180.0;

}

std::vector<std::vector<float>> Track::getPlotVectors() {
	std::vector<std::vector<float>> plotVectors;
	plotVectors.push_back(x_measVec);
	plotVectors.push_back(y_measVec);
	plotVectors.push_back(x_predictVec);
	plotVectors.push_back(y_predictVec);
	return plotVectors;
}