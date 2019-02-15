#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Kalman::Kalman(const Eigen::Vector2f& navDet, const float& vx, const float& vy) {
	
	//TRANSITION MATRIX
	A << 1, dt, 0, 0,
		 0, 1, 0, 0,
		 0, 0, 1, dt,
	   	 0, 0, 0, 1;

	//NOISE EVOLUTION
	G = Eigen::MatrixXf(4, 2);

	G << std::pow(dt, 2) / 2, 0,
		dt, 0,
		0, std::pow(dt, 2) / 2,
		0, dt;

	//STATE OBSERVATION MATRIX
	C = Eigen::MatrixXf(2, 4);
	C << 1, 0, 0, 0,
		 0, 0, 1, 0;

	Q << 10, 0,
		0, 5;
	R << 100, 0,
		0, 100;

	//INITIAL COVARIANCE MATRIX
	P << 10, 0, 0, 0,
		 0, 100, 0, 0,
		 0, 0, 10, 0,
		 0, 0, 0, 100;
	//GAIN     
	K = Eigen::MatrixXf(4, 2);
	
	last_prediction = navDet;
	last_velocity << vx, vy;
	init = true;
}

//Kalman::Kalman(const float dt, const float& x, const float& y, const float& vx, const float& vy, const float& omega, const float& omegad) {
//
//}


void Kalman::predict() {
	if (init) {
		x_predict << last_prediction, last_velocity;
		init = false;
	}
	else {
		//Predicted states based on kalman filtered states
		x_predict = A*x_filter;

		//Predicted covariance matrix
		P_predict = A*P*A.transpose() + G*Q*G.transpose();

		//Predicted measurements
		z_predict = C*x_predict;

		last_prediction << z_predict(0), z_predict(1);
	}
}

void Kalman::gainUpdate() {
	K = P_predict*C.transpose()*(C*P_predict*C.transpose() + R).inverse();
	P = P_predict + K*C*P_predict; //Check correctness
}

void Kalman::update(Eigen::Vector2f& selected_detection) {

	//State update
	x_filter = x_predict+K*(selected_detection-z_predict);
}

Eigen::Vector2f Kalman::getPrediction() {
	return z_predict;
}