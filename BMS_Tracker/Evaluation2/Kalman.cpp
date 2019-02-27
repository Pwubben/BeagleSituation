#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Kalman::Kalman(const Eigen::Vector2d& navDet, const double& vx, const double& vy) {
	
	//TRANSITION MATRIX
	A << 1, dt, 0, 0,
		 0, 1, 0, 0,
		 0, 0, 1, dt,
	   	 0, 0, 0, 1;

	//NOISE EVOLUTION
	G = Eigen::MatrixXd(4, 2);

	G << std::pow(dt, 2) / 2, 0,
		dt, 0,
		0, std::pow(dt, 2) / 2,
		0, dt;

	//STATE OBSERVATION MATRIX
	C = Eigen::MatrixXd(2, 4);
	C << 1, 0, 0, 0,
		 0, 0, 1, 0;

	Q << 20, 0, 0 , 0,
		0, 3, 0 , 0,
		0 ,0, 20, 0,
		0, 0 , 0,3;

	R << 1, 0,
		0, 1;

	//INITIAL COVARIANCE MATRIX
	P << 15, 0, 0, 0,
		 0, 2, 0, 0,
		 0, 0, 15, 0,
		 0, 0, 0,2;
	//GAIN     
	K = Eigen::MatrixXd(4, 2);
	
	last_prediction = navDet;
	last_velocity << vx, vy;
	init = true;
}

//Kalman::Kalman(const double dt, const double& x, const double& y, const double& vx, const double& vy, const double& omega, const double& omegad) {
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
	Eigen::MatrixXd S = (C*P_predict*C.transpose() + R);
	K = P_predict*C.transpose()*S.inverse();
	P = P_predict - K*C*P_predict; 

	//std::cout << "dt: \n" << dt << std::endl;
	//std::cout << "K: \n" << K<< std::endl;
	//std::cout << "P: \n" << P << std::endl;
}

void Kalman::update(Eigen::Vector2d& selected_detection) {
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
			
		}
		//std::cout << "x_filter: \n" << x_filter << std::endl;
	}
}
	

Eigen::Vector2d Kalman::getPrediction() {
	return z_predict;
}

void Kalman::setMatchFlag(int mf) {
	matchFlag = mf;
}

void Kalman::setR(Eigen::MatrixXd R_) {
	R = R_;
}