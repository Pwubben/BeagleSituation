#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

EKF::EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varSpeed, double varYaw)
	: _max_turn_rate(max_turn_rate), _max_acceleration(max_acceleration), _max_yaw_accel(max_yaw_accel)
{
	I = Eigen::MatrixXd::Identity(n, n);
	
	P = Eigen::MatrixXd(n, n);
	P << 100.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 100.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 1000.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 1000.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1000.0;
		

	R = Eigen::MatrixXd(4, 4); //4 sources of measurement for Beagle
	R << pow(varGPS, 2), 0.0, 0.0, 0.0, 0.0,
		0.0, pow(varGPS, 2), 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, pow(varYaw, 2), 0.0,
		0.0, 0.0, pow(varSpeed, 2), 0.0, 0.0;
}

void EKF::compute(Eigen::Vector2f z, double dt) {
	/********************
	- Prediction step
	*********************/
	//Update process covariance
	updateQ(dt);
	//Update state and Jacobian
	updateJA(dt);
	//Prediction
	predict();

	/********************
	- Update step
	*********************/
	//Jacobian Beagle
	Eigen::MatrixXd JH(4, 5);
	JH <<	1.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0;

	//Update parameters
	update(z, x, JH, R);
}

void EKF::updateQ(double dt) {
	Q = Eigen::MatrixXd(n, n);
	sGPS = 0.5 * _max_acceleration * pow(dt, 2);
	sVelocity = _max_acceleration * dt;
	sCourse = _max_turn_rate * dt;
	sYaw = _max_yaw_accel * dt;
	sAccel = _max_acceleration;
	Q << pow(sGPS, 2), 0.0, 0.0, 0.0, 0.0, 0.0,
		0.0, pow(sGPS, 2), 0.0, 0.0, 0.0, 0.0,
		0.0, 0.0, pow(sCourse, 2), 0.0, 0.0, 0.0,
		0.0, 0.0, 0.0, pow(sVelocity, 2), 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, pow(sYaw, 2), 0.0,
		0.0, 0.0, 0.0, 0.0, 0.0, pow(sAccel, 2);

}

void EKF::updateJA(double dt) {
	//Update state
	/***************
	x(0) = x
	x(1) = y
	x(2) = yaw
	x(3) = v
	x(4) = yaw rate
	****************/

	// Updating state equations
	if (fabs(x(4)) < 0.01) {
		x(0) = x(0) + (x(3) * dt) * cos(x(2));
		x(1) = x(1) + (x(3) * dt) * sin(x(2));
		x(2) = x(2);
		x(3) = x(3);
		x(4) = 0.0000001;
	}
	else {
		x(0) = x(0) + (x(3) / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2)));
		x(1) = x(1) + (x(3) / x(4)) * (-cos(x(4) * dt + x(2)) + cos(x(2)));
		x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = x(3) + x(5) * dt;
		x(4) = x(4);
	}

	int n = x.size();
	JA = Eigen::MatrixXd(n, n);

	JA <<	1.0, 0.0, (x(3) / x(4)) * (cos(x(4) * dt + x(2)) - cos(x(2))), (1 / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2))), (x(3) / pow(x(4), 2)) * (-sin(x(4) * dt + x(2)) + x(4)*dt* cos(x(4)*dt + x(2)) + sin(x(2))),
			0.0, 1.0, (x(3) / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2))), (1 / x(4)) * (-cos(x(4) * dt + x(2)) + cos(x(2))), (x(3) / pow(x(4), 2)) * (cos(x(4) * dt + x(2)) + x(4)*dt* sin(x(4)*dt + x(2)) - cos(x(2))),
			0.0, 0.0, 1.0, 0.0, dt,
			0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0;
}

void EKF::predict() {
	// Prediction step
	P = JA * P * JA.transpose() + Q;
}

void EKF::update(const Eigen::VectorXd& Z, const Eigen::VectorXd& Hx, const Eigen::MatrixXd &JH, const Eigen::MatrixXd &R) {
	Eigen::MatrixXd JHT = P * JH.transpose();
	// Temporary variable for storing this intermediate value
	Eigen::MatrixXd S = JH * JHT + R;
	// Compute the Kalman gain
	K = JHT * S.inverse();

	// Update estimate
	x = x + K * (Z - Hx);
	// Update the error covariance
	P = (I - K * JH) * P;
}