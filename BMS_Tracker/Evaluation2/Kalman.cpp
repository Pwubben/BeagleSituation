#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Kalman::Kalman(const Eigen::Vector2d& navDet, const double& v, const double& heading, const Eigen::Matrix4d& Q_, const Eigen::Matrix4d& P_, const int& modelNumber = 0) :
	Q(Q_),
	P(P_),
	A(getDynamicModel(modelNumber)),
	model(modelNumber)
{
	//STATE OBSERVATION MATRIX
	C = Eigen::MatrixXd(2, 4);
	C << 1, 0, 0, 0,
		 0, 0, 1, 0;

	//GAIN     
	K = Eigen::MatrixXd(4, 2);
	
	last_prediction = navDet;
	last_velocity << sin(heading)*v, cos(heading)*v;
	init = true;
}

void Kalman::compute(Eigen::VectorXd detection)
{
	//Target processing
	//Track prediction
	if (matchFlag < 2) {

		//KF gain update
		gainUpdate();
		//Track update

		update(detection);

		predict();
	}
	else {

		update(detection);

		predict();
	}
}

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

void Kalman::update(Eigen::VectorXd& selected_detection) {
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
Eigen::MatrixXd Kalman::getDynamicModel(int modelNumber) {
	std::vector<Eigen::MatrixXd> models;
	double numStates = 4;

	//Variable which is used to temporarily store the dynamic models
	Eigen::MatrixXd Model(numStates, numStates);

	//Model 0 - Constant Velocity
	Model << 1, dt, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, dt,
		0, 0, 0, 1;

	models.push_back(Model);
	modelTurnRates.push_back(0.0);
	//Model 1 - Constant Turn - 30 deg/s

	double omega = Util::deg2Rad(30);

	
	Model << 1, 1 / omega * sin(omega*dt), 0, -1 / omega * (1 - cos(omega*dt)),
		0, cos(omega*dt), 0, -sin(omega*dt),
		0, 1 / omega * (1 - cos(omega*dt)), 1, 1 / omega * sin(omega*dt),
		0, sin(omega*dt), 0, cos(omega*dt);

	models.push_back(Model);
	modelTurnRates.push_back(omega);

	double omega = Util::deg2Rad(-30);

	Model << 1, 1 / omega * sin(omega*dt), 0, -1 / omega * (1 - cos(omega*dt)),
		0, cos(omega*dt), 0, -sin(omega*dt),
		0, 1 / omega * (1 - cos(omega*dt)), 1, 1 / omega * sin(omega*dt),
		0, sin(omega*dt), 0, cos(omega*dt);

	models.push_back(Model);
	modelTurnRates.push_back(omega);

	//Model 3 - Constant Turn - 50 deg/s

	double omega = Util::deg2Rad(50);

	Model << 1, 1 / omega * sin(omega*dt), 0, -1 / omega * (1 - cos(omega*dt)),
		0, cos(omega*dt), 0, -sin(omega*dt),
		0, 1 / omega * (1 - cos(omega*dt)), 1, 1 / omega * sin(omega*dt),
		0, sin(omega*dt), 0, cos(omega*dt);

	models.push_back(Model);
	modelTurnRates.push_back(omega);

	double omega = Util::deg2Rad(-50);

	Model << 1, 1 / omega * sin(omega*dt), 0, -1 / omega * (1 - cos(omega*dt)),
		0, cos(omega*dt), 0, -sin(omega*dt),
		0, 1 / omega * (1 - cos(omega*dt)), 1, 1 / omega * sin(omega*dt),
		0, sin(omega*dt), 0, cos(omega*dt);

	models.push_back(Model);
	modelTurnRates.push_back(omega);

	return models[modelNumber];
}

Eigen::Vector2d Kalman::getPrediction() {
	return z_predict;
}

Eigen::VectorXd Kalman::getState()
{
	Eigen::VectorXd stateExtend(5);
	//TODO-Check atan2 angle
	stateExtend << x_predict(0), x_predict(2), sqrt(pow(x_predict(1), 2) + pow(x_predict(3), 2)), atan2(x_predict(1), x_predict(3)), modelTurnRates[model];
	return stateExtend;
}

void Kalman::setState(Eigen::VectorXd x_mixed) {
	double xv, yv;

	//Convert IMM states to linear states
	xv = sin(x_mixed(3))*x_mixed(2);
	yv = cos(x_mixed(3))*x_mixed(2);

	x_predict << x_mixed(0), xv, x_mixed(1), yv;
}

Eigen::MatrixXd Kalman::getCovariance()
{
	return P_predict;
}

double Kalman::getProbability()
{
	return lambda;
}

void Kalman::setMatchFlag(int mf) {
	matchFlag = mf;
}

void Kalman::setR(Eigen::MatrixXd R_) {
	R = R_;
}

void Kalman::setCovariance(Eigen::MatrixXd P_) {
	P = P_;
}