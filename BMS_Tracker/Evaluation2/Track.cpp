#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

void Track::run() {
	//KF gain update

	//Track update

	//Track prediction
}

prediction Track::getPrediction() {
	return prediction_;
}

double Track::getDetection() {
	return range_;
}

void Track::setDetection(double range, double angle) {
	range_ = range;
	angle_ = angle;
	//Compute detection in navigation frame coordinates
	body2nav();
}

cv::Point Track::body2nav() {
	cv::Point relDet = cv::Point(sin(angle_ / 180 * M_PI)*range_, cos(angle_ / 180 * M_PI)*range_);
	double rotMat[2][2] = { {cos(heading) , -sin(heading)},
							{sin(heading) ,  cos(heading)} };
	//Compute detection in navigation frame coordinates
	navDet = cv::Point(rotMat[1][1] * relDet.x + rotMat[1][2] * relDet.y, rotMat[2][1] * relDet.x + rotMat[2][2] * relDet.y);
}