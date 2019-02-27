
#include "Tracker.h"
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

EKF::EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varYaw, double varYawRate, Eigen::VectorXd xInit)
	: dt(1/double(15)), init(true), _max_turn_rate(max_turn_rate), _max_acceleration(max_acceleration), _max_yaw_accel(max_yaw_accel), x(xInit)
{
	I = Eigen::MatrixXd::Identity(n, n);
	
	P = Eigen::MatrixXd(n, n);
	P << 10.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 10.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 10.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 15.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 10.0;
		

	R = Eigen::MatrixXd(4, 4); //4 sources of measurement for Beagle
	R << pow(varGPS, 2), 0.0, 0.0, 0.0,
		0.0, pow(varGPS, 2), 0.0, 0.0,
		0.0, 0.0, pow(varYaw, 2), 0.0,
		0.0, 0.0, 0.0, pow(varYawRate, 2);

	//Jacobian Beagle
	JH = Eigen::MatrixXd(4, 5);
	JH << 1.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 1.0;
}

void EKF::compute(Eigen::VectorXd z) {
	if (init) {
		
		init = false;
	}
	else {
		
		//std::cout << "z: \n" << z << std::endl;
		/********************
		- Update step
		*********************/
		

		//Eigen::VectorXf Hx(z.size());
		//Hx << x(0), x(1), x(2), x(4);
		//Update parameters
		//update(z, JH*x, JH, R);

		/********************
		- Prediction step
		*********************/
		//Update process covariance
		updateQ(dt);
		//Update state and Jacobian
		updateJA(dt);
		//Prediction
		predict();
	}
}

void EKF::updateQ(double dt) {
	Q = Eigen::MatrixXd(n, n);
	sGPS = 0.5 * _max_acceleration * pow(dt, 2);
	sVelocity = _max_acceleration * dt;
	sCourse = _max_turn_rate * dt;
	sYaw = _max_yaw_accel * dt;
	sAccel = _max_acceleration;
	Q << pow(sGPS, 2), 0.0, 0.0, 0.0, 0.0,
		0.0, pow(sGPS, 2), 0.0, 0.0, 0.0,
		0.0, 0.0, pow(sCourse, 2), 0.0, 0.0,
		0.0, 0.0, 0.0, pow(sVelocity, 2), 0.0,
		0.0, 0.0, 0.0, 0.0, pow(sYaw, 2);
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
	//std::cout <<"x:\n"<< x << std::endl;
	// Updating state equations
	//std::cout << "dt: " << dt << std::endl;
	if (fabs(x(4)) < 0.01) {
		x(0) = x(0) + (x(3) * dt) * sin(x(2));
		x(1) = x(1) + (x(3) * dt) * cos(x(2));
		x(2) = std::fmod((x(2) + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = x(3);
		x(4) = Util::sgn(x(4))*std::max(double(abs(x(4))),double(0.0001));
	}
	else {
		x(0) = x(0) + (x(3) / x(4)) * (-cos(x(4) * dt + x(2)) + cos(x(2)));
		x(1) = x(1) + (x(3) / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2)));
		x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = x(3);
		x(4) = x(4);
	}

	//Wrap angles
	//if (x(2) < 0)
	//	x(2) += 2.0*M_PI;

	//std::cout <<"x state update: \n"<< x << std::endl;
	int n = x.size();
	JA = Eigen::MatrixXd(n, n);
	if (fabs(x(4)) < 0.01) {
		JA << 1.0, 0.0, x(3)*dt*cos(x(2)), dt*sin(x(2)), 0,
			0.0, 1.0, -x(3)*dt*sin(x(2)), dt*cos(x(2)), 0,
			0.0, 0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0;
	}
	else {
		JA << 1.0, 0.0, (x(3) / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2))), (1.0 / x(4)) * (-cos(x(4) * dt + x(2)) + cos(x(2))), (x(3) / pow(x(4), 2)) * (-cos(x(4) * dt + x(2)) - x(4)*dt* sin(x(4)*dt + x(2)) + cos(x(2))),
			0.0, 1.0, (x(3) / x(4)) * (cos(x(4) * dt + x(2)) - cos(x(2))), (1.0 / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2))), (x(3) / pow(x(4), 2)) * (-sin(x(4) * dt + x(2)) + x(4)*dt* cos(x(4)*dt + x(2)) + sin(x(2))),
			0.0, 0.0, 1.0, 0.0, dt,
			0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 1.0;
	}
}

void EKF::predict() {
	// Prediction step
	P = JA * P * JA.transpose() + Q;
}

void EKF::update(const Eigen::VectorXd& z) {

	if (init) {
		if (z.size() == 4) {
			//Beagle initiation
			x(0) = z(0);
			x(1) = z(1);
			x(2) = z(2);
			x(4) = z(3);
		}
	}
	else {
		//std::cout << "x: " << z(0) << "- y: " << z(1) << std::endl;


		Eigen::MatrixXd JHT = P * JH.transpose();
		// Temporary variable for storing this intermediate value
		Eigen::MatrixXd S = JH * JHT + R;
		// Compute the Kalman gain
		K = JHT * S.inverse();

		// Update estimate
		x = x + K * (z - JH*x);

		x_predictVec.push_back(x(0));
		y_predictVec.push_back(x(1));
		x_measVec.push_back(z(0));
		y_measVec.push_back(z(1));

		//std::cout << "Z: \n" <<Z - Hx << std::endl;
		//std::cout << "x_update: \n" << x << std::endl;
		//std::cout << "z: \n" << Z << std::endl;
		//std::cout << "JA: \n" << JA << std::endl;
		//std::cout << "K:\n" << K << std::endl;
		//std::cout << "R:\n" << R << std::endl;
		// Update the error covariance
		P = (I - K * JH) * P;
	}
}

Eigen::Vector3d EKF::getBeaglePrediction() {
	Eigen::Vector3d prediction;
	//beaglePrediction prediction;
	prediction << x(0), x(1), x(2);
	return prediction;
}

std::vector<std::vector<double>> EKF::getPlotVectors() {
	std::vector<std::vector<double>> plotVectors;
	plotVectors.push_back(x_measVec);
	plotVectors.push_back(y_measVec);
	plotVectors.push_back(x_predictVec);
	plotVectors.push_back(y_predictVec);
	return plotVectors;
}


