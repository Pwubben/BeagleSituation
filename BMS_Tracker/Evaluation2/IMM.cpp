#include "Tracker.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

IMM::IMM(const int& modelNum, const std::vector<int>& modelNumbers, const std::vector<Eigen::MatrixXd>& Q_, const std::vector<Eigen::MatrixXd>& P_, const Eigen::Vector2d& navDet, const double& vx, const double& vy) : P_model(modelNum, Eigen::MatrixXd(numStates, numStates)), P_mixed(modelNum, Eigen::MatrixXd(numStates, numStates)) {
	
	//State interaction variables
	x_model = Eigen::MatrixXd(numStates,modelNum) ;
	x_mixed = Eigen::MatrixXd(numStates, modelNum);
	mu_tilde = Eigen::MatrixXd(modelNum, modelNum);

	stateTransitionProb = Eigen::MatrixXd(modelNum, modelNum);

	//Model probability update
	lambda = Eigen::VectorXd(modelNum);
	mu_hat = Eigen::VectorXd(modelNum);

	//State estimate combination
	x = Eigen::VectorXd(numStates);
	P = Eigen::MatrixXd(numStates,numStates);

	for (auto &&No : modelNumbers) {
		if (No < 5)
			filters.emplace_back(new Kalman(navDet, v, heading, Q_[No], P_[No], No));
		else
			filters.emplace_back(new EKF(navDet, v, heading, Q_[No], P_[No], No));
	}
}

void IMM::run(Eigen::VectorXd z)
{
	stateInteraction();

	for (auto &&f : filters) {
		filters->setR(Rvec[f]);
		filters->compute(z);

		//Retrieve information from filters
		lambda(f) = filters[f]->getProbability();
		x_model.col(f) = filters[f]->getState();
		P_model[f] = filters[f]->getCovariance();

		P_mixed[f] = P_mixed[f].setZero();
	}

	P.setZero();

	modelProbabilityUpdate();

	stateEstimateCombination()
}

void IMM::stateInteraction()
{
	double cBar;
	for (int i = 0; i < filters.size(); i++) {
		cBar = stateTransitionProb.col(i).transpose()*mu_hat;
		for (int j = 0; j < filters.size(); j++) {
			mu_tilde(i, j) = 1/cBar* stateTransitionProb(i, j) * mu_hat(i);
		}
	}

	x_mixed = x_model * mu_tilde;

	for (int j = 0; j < filters.size(); j++) {
		for (int i = 0; i < filters.size(); i++) {
			P_mixed[j] += mu_tilde(i, j) * (P_model[i] + (x_model.col(i) - x_mixed.col(i))*(x_model.col(i) - x_mixed.col(i)).transpose());
		}
	}
}

void IMM::modelProbabilityUpdate()
{
	//Normalization factor
	double c = lambda.sum();
	
	//Model probability
	mu_hat = 1 / c * lambda;
}

void IMM::stateEstimateCombination()
{
	x = x_model * mu_hat;

	for (int i = 0; i < filters.size(); i++) {
			P += mu_hat(i) * (P_model[i] + (x_model.col(i) - x.col(i))*(x_model.col(i) - x.col(i)).transpose());
	}
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

