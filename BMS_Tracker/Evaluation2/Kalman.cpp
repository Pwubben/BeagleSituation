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

	Q << 1, 0, 0 , 0,
		0, 0.1, 0 , 0,
		0 ,0, 1, 0,
		0, 0 , 0, 0.1;

	R << 1, 0,
		0, 1;

	//INITIAL COVARIANCE MATRIX
	P << 1, 0, 0, 0,
		 0, 0.1, 0, 0,
		 0, 0, 1, 0,
		 0, 0, 0,0.1;
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

		//Predicted states based on kalman filtered states
		x_predict = A*x_filter;	
	
		//Predicted covariance matrix
		
		//Predicted measurements
		z_predict = C*x_predict;
		//std::cout <<"X_predict:\n"<< x_predict << std::endl;
		last_prediction << z_predict(0), z_predict(1);
}

void Kalman::gainUpdate() {
	P_predict = A*P*A.transpose() + Q;
	Eigen::MatrixXf S = (C*P_predict*C.transpose() + R);
	K = P_predict*C.transpose()*S.inverse();
	P = P_predict - K*C*P_predict; 

	//std::cout << "dt: \n" << dt << std::endl;
	//std::cout << "K: \n" << K<< std::endl;
	//std::cout << "P: \n" << P << std::endl;
}

void Kalman::update(Eigen::Vector2f& selected_detection) {
	if (init) {
		x_filter << last_prediction(0), last_velocity(0), last_prediction(1), last_velocity(1);
		init = false;
	}
	else {
		if (matchFlag == 2)
			x_filter = x_predict;
		else
		{
			x_filter = x_predict + K*(selected_detection - z_predict);
			//State update
			//std::cout << "Error: \n" << selected_detection - z_predict << std::endl;
			std::cout << "x_filter: \n" << x_filter << std::endl;
		}
	}
}
	

Eigen::Vector2f Kalman::getPrediction() {
	return z_predict;
}

void Kalman::setMatchFlag(int mf) {
	matchFlag = mf;
}

void Kalman::setR(Eigen::MatrixXf R_) {
	R = R_;
}