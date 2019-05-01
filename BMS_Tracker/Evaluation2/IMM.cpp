#include "Tracker.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

IMM::IMM(const int& modelNum, const std::vector<int>& modelNumbers, const std::vector<Eigen::MatrixXd>& Q_, const std::vector<Eigen::MatrixXd>& P_, const Eigen::Vector2d& navDet, const double& vInit, const double& headingInit, Eigen::Vector3d beagleMeas) : 
	P_model(modelNum, Eigen::MatrixXd(numStates, numStates)), 
	P_mixed(modelNum, Eigen::MatrixXd(numStates, numStates)) 
{
	
	//State interaction variables
	x_model = Eigen::MatrixXd(numStates,modelNum);
	x_mixed = Eigen::MatrixXd(numStates, modelNum);
	mu_tilde = Eigen::MatrixXd(modelNum, modelNum);

	stateTransitionProb = Eigen::MatrixXd(modelNum, modelNum);

	//Model probability update
	lambda = Eigen::VectorXd(modelNum);
	mu_hat = Eigen::VectorXd(modelNum);
	for (int i = 0; i < mu_hat.size(); i++)
		mu_hat(i) = 1.0 / double(modelNum);

	//State estimate combination
	x = Eigen::VectorXd(numStates);
	P = Eigen::MatrixXd(numStates,numStates);
	
	std::cout << Q_[1] << std::endl;
	//filters.emplace_back(std::shared_ptr<KalmanFilters>(new Kalman(navDet, vInit, headingInit, Q_[1], P_[1], 1)));
	for (auto &&No : modelNumbers) {
		if (No < 5) 
			filters.push_back(std::unique_ptr<KalmanFilters>(new Kalman(navDet, vInit, headingInit, Q_[0], P_[0], No)));
		else
			filters.push_back(std::unique_ptr<KalmanFilters>(new EKF(navDet, vInit, headingInit, Q_[1], P_[1], No, beagleMeas)));
	}
}

void IMM::run(Eigen::VectorXd z, double radVel, double angle_, double omega, Eigen::VectorXd beagleMeas)
{
	if (init)
		init = false;
	else {
		if (matchFlag == 0) {
			stateInteraction();
			if (matchFlag == 0) {
				for (int i = 0; i < filters.size(); i++) {
					filters[i]->setState(x_mixed.col(i));
					filters[i]->setCovariance(P_mixed[i]);
				}
			}
		}
	}

#pragma omp parallel for 
	for (int i = 0; i < filters.size(); i++) {
		filters[i]->setMatchFlag(matchFlag);
		filters[i]->setR(Rvec[i]);
		filters[i]->compute(z, radVel, angle_, omega, beagleMeas);
		
		//Retrieve information from filters
		lambda(i) = filters[i]->getProbability();
		x_model.col(i) = filters[i]->getState();
		P_model[i] = filters[i]->getCovariance();
		//std::cout << "P: " << i << "\n" << P_model[i] << std::endl;
		P_mixed[i] = P_mixed[i].setZero();
		//std::cout << lambda << std::endl;
	}
	//if (matchFlag == 0)
		//std::cout <<"lambda:\n "<<lambda << std::endl;
	if (matchFlag == 0) 
		modelProbabilityUpdate();

	P.setZero();

	stateEstimateCombination();
	
}

void IMM::stateInteraction()
{
	double cBar;
	for (int j = 0; j < filters.size(); j++) {
		cBar = stateTransitionProb.col(j).transpose()*mu_hat;
		for (int i = 0; i < filters.size(); i++) {
			mu_tilde(i, j) = 1.0/cBar* stateTransitionProb(i, j) * mu_hat(i);
			//std::cout << "stateprob: \n" << stateTransitionProb(i, j) << std::endl;
		}
	}

	//std::cout << "c_bar: \n" << cBar << std::endl;
	//std::cout << "mu_tilde: \n" << mu_tilde << std::endl;
	//std::cout << "x_model: \n" << x_model << std::endl;
	x_mixed = x_model * mu_tilde;
	//std::cout << "x_mixed: \n" << x_mixed.transpose() << std::endl;

	for (int j = 0; j < filters.size(); j++) {
		for (int i = 0; i < filters.size(); i++) {
			P_mixed[j] += mu_tilde(i, j) * (P_model[i] + (x_model.col(i) - x_mixed.col(j))*(x_model.col(i) - x_mixed.col(j)).transpose());
			//std::cout << "P: " << j << "\n " << P_mixed[j] << std::endl;
		}
	}
}

void IMM::modelProbabilityUpdate()
{
	//Normalization factor
	double c = lambda.sum();
	
	//Model probability
	mu_hat = 1.0 / c * lambda;
	/*if (matchFlag == 0)
		std::cout << "mu_hat:\n" << mu_hat << std::endl;*/
}

void IMM::stateEstimateCombination()
{
	
	if (matchFlag < 2) {
		x = x_model * mu_hat;
		//std::cout << "x_model: \n" << x_model << std::endl;
		//std::cout << "x_end: \n" << x.transpose() << std::endl;
		//std::cout << "mu_hat: " << mu_hat.transpose() << std::endl;
		//std::cout << "Lambda: " << lambda.transpose() << std::endl;
	}
	else {
		//if (fabs(x(4)) < 0.01) {
		//	x(0) = x(0) + (x(3) * dt) * sin(x(2));
		//	x(1) = x(1) + (x(3) * dt) * cos(x(2));
		//	x(2) = std::fmod((x(2) + M_PI), (2.0 * M_PI)) - M_PI;
		//	x(3) = x(3);
		//	x(4) = Util::sgn(x(4))*std::max(double(abs(x(4))), double(0.0001));
		//}
		//else {
		//	x(0) = x(0) + (x(3) / x(4)) * (-cos(x(4) * dt + x(2)) + cos(x(2)));
		//	x(1) = x(1) + (x(3) / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2)));
		//	x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
		//	x(3) = x(3);
		//	x(4) = x(4);
		//}
		double L = 25;
		double V = 56;
		double D = 0.0222 / sqrt(9.81*pow(V, 1.0 / 3.0))*pow(L / pow(V, 1.0 / 3.0), 2.85);
		if (fabs(x(4)) < 0.005) {
			x(0) = x(0) + (x(3) * dt) * sin(x(2));
			x(1) = x(1) + (x(3) * dt) * cos(x(2));
			x(2) = std::fmod((x(2) + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = Util::sgn(x(4))*std::max(double(abs(x(4))), double(0.0001));
		}
		else {
			x(0) = x(0) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*x(4))) * (-cos(x(3)*dt / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) + x(2)) + cos(x(2)));
			x(1) = x(1) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*x(4))) * (sin(x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) * dt + x(2)) - sin(x(2)));
			x(2) = std::fmod((x(2) + x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) * dt + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = Util::sgn(x(4))*std::min(double(abs(x(4))), double(60));
		}
	}
	for (int i = 0; i < filters.size(); i++) {
			P += mu_hat(i) * (P_model[i] + (x_model.col(i) - x)*(x_model.col(i) - x).transpose());
			//std::cout << P << std::endl;
	}
}

Eigen::Vector2d IMM::getPrediction() {
	return x.head(2);
}

Eigen::VectorXd IMM::getState() {
	return x;
}

Eigen::VectorXd IMM::getMu() {
	return mu_hat;
}

void IMM::setStateTransitionProbability(Eigen::MatrixXd stateTransitionProb_) {
	stateTransitionProb = stateTransitionProb_;
}

void IMM::setR(std::vector<Eigen::MatrixXd> Rvec_) {
	Rvec = Rvec_;
}

void IMM::setMatchFlag(int mf) {
	matchFlag = mf;
}

