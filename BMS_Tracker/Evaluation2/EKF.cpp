
#include "Tracker.h"
#include <algorithm>
#include <cmath>
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include <random>
EKF::EKF(double max_acceleration, double max_turn_rate, double max_yaw_accel, double varGPS, double varYaw, double varYawRate, Eigen::VectorXd xInit)
	: dt(1/double(15)), init(true), _max_turn_rate(max_turn_rate), _max_acceleration(max_acceleration), _max_yaw_accel(max_yaw_accel), x(xInit), BeagleObject(true)
{
	I = Eigen::MatrixXd::Identity(n, n);
	
	P = Eigen::MatrixXd(n, n);
	P << 10.0, 0.0, 0.0, 0.0, 0.0, 
		0.0, 10.0, 0.0, 0.0, 0.0, 
		0.0, 0.0, 10.0, 0.0, 0.0, 
		0.0, 0.0, 0.0, 15.0, 0.0,
		0.0, 0.0, 0.0, 0.0, 10.0;
		
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

EKF::EKF(const Eigen::Vector2d& navDet, const double& v, const double& heading, const Eigen::MatrixXd& Q_, const Eigen::MatrixXd& P_, const int& modelNumber, Eigen::Vector3d beagleMeas) :
	Q(Q_),
	P(P_),
	dt(1 / double(15)),
	modelNum(modelNumber)
{
	I = Eigen::MatrixXd::Identity(n, n);

	//Initial state
	x = Eigen::VectorXd(5);
	x << navDet, heading, v, 0;
	/*std::cout << "Beaglemeas: " << beagleMeas << std::endl;
	std::cout << "nav: " << navDet.transpose() << std::endl;
	std::cout << "x: " << x.transpose() << std::endl;*/
	JH = Eigen::MatrixXd(2, 5);
	JH << 1.0, 0.0, 0.0, 0.0, 0.0,
		  0.0, 1.0, 0.0, 0.0, 0.0;
}

void EKF::compute(Eigen::VectorXd z, double radVel_, double angle_, Eigen::Vector3d beagleMeas) {
	if (init) {
		init = false;
	}
	else {
		
		/********************
		- Prediction step
		*********************/
		if (BeagleObject) {

			//Update state and Jacobian
			updateJA(dt);
			//Prediction
			predict();
		}
		else {
			if (matchFlag < 2) {
				Eigen::VectorXd Z_(3);
				if (matchFlag == 0) {
					Z_.resize(3);
					Z_ << z, radVel_;
				}
				else {
					Z_.resize(2);
					Z_ << z;
				}
				update(Z_, angle_,beagleMeas);
				updateJA(dt);
				predict();
			}
			else {
				updateJA(dt);
			}
		}
	}
}


void EKF::predict() {
	// Prediction step
	P = JA * P * JA.transpose() + Q;
}

void EKF::update(const Eigen::VectorXd& z, double angle_, Eigen::Vector3d beagleMeas) {

	if (init) {
		if (BeagleObject) {
			//Beagle initiation
			x(0) = z(0);
			x(1) = z(1);
			x(2) = z(2);
			x(4) = z(3);
		}
		
	}
	else {
		if (!BeagleObject) {
			if (R.cols() == 2) {
				JH.resize(2, 5);
				JH << 1.0, 0.0, 0.0, 0.0, 0.0,
					0.0, 1.0, 0.0, 0.0, 0.0;
				Hx.resize(2);
				Hx = JH*x;
			}
			else {
				JH.resize(3, 5);
				JH << 1.0, 0.0, 0.0, 0.0, 0.0,
					  0.0, 1.0, 0.0, 0.0, 0.0,
					  0.0, 0.0, x(3)*sin(M_PI + angle_ - x(2)), cos(M_PI + angle_ - x(2)), 0.0;
				Hx.resize(3);
				Hx << x(0), x(1), x(3)*cos(M_PI + angle_ - x(2));
			}
		}
		//if (!BeagleObject) {
		//	if (R.cols() == 1) {
		//		JH.resize(1, 5);
		//		JH <<	(x(1) - beagleMeas(1)) / double(pow(beagleMeas(0), 2) - 2.0*beagleMeas(0)*x(0) + pow(beagleMeas(1), 2) - 2.0*beagleMeas(1)*x(1) + pow(x(0), 2) + pow(x(1), 2)), (beagleMeas(1) - x(0)) / double(pow(beagleMeas(0), 2) - 2.0*beagleMeas(0)*x(0) + pow(beagleMeas(1), 2) - 2.0*beagleMeas(1)*x(1) + pow(x(0), 2) + pow(x(1), 2)), 0.0, 0.0, 0.0;
		//		Hx.resize(1);
		//		Hx << atan2(x(0)-beagleMeas(0),x(1)-beagleMeas(1))-beagleMeas(2);
		//	}
		//	else {
		//		JH.resize(3, 5);
		//		JH << (x(0) - beagleMeas(0)) / double(sqrt(pow(x(0) - beagleMeas(0), 2) + pow(x(1) - beagleMeas(1), 2))), (x(1) - beagleMeas(1)) / double(sqrt(pow(x(0) - beagleMeas(0), 2) + pow(x(1) - beagleMeas(1), 2))), 0.0, 0.0, 0.0,
		//			(x(1)-beagleMeas(1))/double(pow(beagleMeas(0),2)-2.0*beagleMeas(0)*x(0)+pow(beagleMeas(1),2)-2.0*beagleMeas(1)*x(1)+pow(x(0),2)+pow(x(1),2)) , (beagleMeas(1)- x(0)) / double(pow(beagleMeas(0), 2) - 2.0*beagleMeas(0)*x(0) + pow(beagleMeas(1), 2) - 2.0*beagleMeas(1)*x(1) + pow(x(0), 2) + pow(x(1), 2)), 0.0, 0.0, 0.0,
		//			0.0, 0.0, x(3)*sin(M_PI + angle_ - x(2)), cos(M_PI + angle_ - x(2)), 0.0;
		//		Hx.resize(3);
		//		Hx << sqrt(pow(x(0) - beagleMeas(0), 2) + pow(x(1) - beagleMeas(1), 2)), atan2(x(0) - beagleMeas(0), x(1) - beagleMeas(1)) - beagleMeas(2), x(3)*cos(M_PI + angle_ - x(2));
		//	}
		//}
		
		//std::cout << "x: " << z(0) << "- y: " << z(1) << std::endl;
		if (BeagleObject)
			Hx = JH*x;

		Eigen::MatrixXd JHT = P * JH.transpose();
		// Temporary variable for storing this intermediate value
		Eigen::MatrixXd S = JH * JHT + R;
		// Compute the Kalman gain
		K = JHT * S.inverse();
		
		
		Eigen::VectorXd Z = z - Hx;

		//IMM probability
		if (!BeagleObject)
			lambda = modelProbability(P, R, z);

		if (!BeagleObject) {
			//std::cout << "zm: \n" << z.transpose() << std::endl;
			//std::cout << "z: \n" << Z.transpose() << std::endl;
			//std::cout << "S:\n" << S << std::endl;
			//std::cout << "K: \n" << K << std::endl;
			//std::cout << "x state update: \n" << x << std::endl;
			//std::cout << "JH: \n" << JH << std::endl;
			//std::cout << "Hx: \n" << Hx << std::endl;
		}
		// Update estimate
		x = x + K * (Z);

		if (BeagleObject) {
			x_predictVec.push_back(x(0));
			y_predictVec.push_back(x(1));
			x_measVec.push_back(z(0));
			y_measVec.push_back(z(1));
		}
		//std::cout << "Z: \n" <<Z - Hx << std::endl;

		//std::cout << "x_update: \n" << x << std::endl;
		//std::cout << "z: \n" << Z << std::endl;
		//std::cout << "JA: \n" << JA << std::endl;
		//std::cout << "K:\n" << K << std::endl;
		//std::cout << "R:\n" << R << std::endl;

		if (!BeagleObject ) {
			//std::cout << "x:\n" << x << std::endl;
			//std::cout << "Z:\n" << Z << std::endl;
			//std::cout << "K:\n" << K << std::endl;
			//std::cout << "R:\n" << R << std::endl;
			//std::cout << "S:\n" << S << std::endl;
			//std::cout << "Lambda: " << lambda << std::endl;
		}
		// Update the error covariance
		P = (I - K * JH) * P;
	}
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

	//std::cout << "x state update1: \n" << x << std::endl;
	if (modelNum == 5 || BeagleObject) {
		if (fabs(x(4)) < 0.005) {
			x(0) = x(0) + (x(3) * dt) * sin(x(2));
			x(1) = x(1) + (x(3) * dt) * cos(x(2));
			x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = Util::sgn(x(4))*std::max(double(abs(x(4))), double(0.0001));
		}
		else {
			x(0) = x(0) + (x(3) / x(4)) * (-cos(x(4) * dt + x(2)) + cos(x(2)));
			x(1) = x(1) + (x(3) / x(4)) * (sin(x(4) * dt + x(2)) - sin(x(2)));
			x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = x(4);
		}


		int n = x.size();
		JA = Eigen::MatrixXd(n, n);
		if (fabs(x(4)) < 0.01) {
			JA << 1.0, 0.0, x(3)*dt*cos(x(2)), dt*sin(x(2)), 0,
				0.0, 1.0, -x(3)*dt*sin(x(2)), dt*cos(x(2)), 0,
				0.0, 0.0, 1.0, 0.0, dt,
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
	if (modelNum == 6) {
		x(0) = x(0) + (x(3) * dt) * sin(x(2));
		x(1) = x(1) + (x(3) * dt) * cos(x(2));
		x(2) = std::fmod((x(2) + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = abs(x(3));
		x(4) = 0;

		JA = Eigen::MatrixXd(x.size() , x.size());
		JA << 1.0, 0.0, x(3)*dt*cos(x(2)), dt*sin(x(2)), 0,
				0.0, 1.0, -x(3)*dt*sin(x(2)), dt*cos(x(2)), 0,
				0.0, 0.0, 1.0, 0.0, 0.0,
				0.0, 0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0;
	}
	if (modelNum == 7) {
		double omega = Util::deg2Rad(-3);
		x(0) = x(0) + (x(3) / omega) * (-cos(omega * dt + x(2)) + cos(x(2)));
		x(1) = x(1) + (x(3) / omega) * (sin(omega * dt + x(2)) - sin(x(2)));
		x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = abs(x(3));
		x(4) = omega;

		JA = Eigen::MatrixXd(x.size(), x.size());
		JA << 1.0, 0.0, (x(3) / omega) * (sin(omega * dt + x(2)) - sin(x(2))), (1.0 / omega) * (-cos(omega * dt + x(2)) + cos(x(2))), (x(3) / pow(omega, 2)) * (-cos(omega * dt + x(2)) - x(4)*dt* sin(omega*dt + x(2)) + cos(x(2))),
			0.0, 1.0, (x(3) / omega) * (cos(omega * dt + x(2)) - cos(x(2))), (1.0 / omega) * (sin(omega * dt + x(2)) - sin(x(2))), (x(3) / pow(omega, 2)) * (-sin(omega * dt + x(2)) + omega*dt* cos(omega*dt + x(2)) + sin(x(2))),
			0.0, 0.0, 1.0, 0.0, dt,
			0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0;
	}
	if (modelNum == 8) {
		double omega = Util::deg2Rad(3);
		x(0) = x(0) + (x(3) / omega) * (-cos(omega * dt + x(2)) + cos(x(2)));
		x(1) = x(1) + (x(3) / omega) * (sin(omega * dt + x(2)) - sin(x(2)));
		x(2) = std::fmod((x(2) + x(4) * dt + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = abs(x(3));
		x(4) = omega;

		JA = Eigen::MatrixXd(x.size(), x.size());
		JA << 1.0, 0.0, (x(3) / omega) * (sin(omega * dt + x(2)) - sin(x(2))), (1.0 / omega) * (-cos(omega * dt + x(2)) + cos(x(2))), (x(3) / pow(omega, 2)) * (-cos(omega * dt + x(2)) - x(4)*dt* sin(omega*dt + x(2)) + cos(x(2))),
			0.0, 1.0, (x(3) / omega) * (cos(omega * dt + x(2)) - cos(x(2))), (1.0 / omega) * (sin(omega * dt + x(2)) - sin(x(2))), (x(3) / pow(omega, 2)) * (-sin(omega * dt + x(2)) + omega*dt* cos(omega*dt + x(2)) + sin(x(2))),
			0.0, 0.0, 1.0, 0.0, dt,
			0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0;
	}
}

double EKF::modelProbability(Eigen::MatrixXd P, Eigen::MatrixXd R, const Eigen::VectorXd& z) {
	Eigen::MatrixXd JHProb(2, 5);
	JHProb << 1.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0, 0.0;

	Eigen::MatrixXd JHT = P * JH.transpose();
	Eigen::MatrixXd S = JH * JHT + R;

	double lambda = 1 / (2 * M_PI * sqrt(S.determinant())) * std::exp(-0.5*(z-Hx).transpose()*S.inverse()*(z - Hx));

	if (lambda < 1e-30)
		lambda = 1e-30;

	return lambda;
}

Eigen::Vector2d EKF::getPrediction() {

	return x.head(2);
}

void EKF::setR(Eigen::MatrixXd R_) {

	R = R_;
}

Eigen::Vector4d EKF::getBeaglePrediction() {
	Eigen::Vector4d prediction;
	//beaglePrediction prediction;
	prediction << x(0), x(1), x(2), x(3);
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


double EKF::getProbability() {
	return lambda;
}

Eigen::VectorXd EKF::getState() {
	return x;
}

Eigen::MatrixXd EKF::getCovariance() {
	return P;
}

void EKF::setState(Eigen::VectorXd x_mixed) {
	x = x_mixed;
}

void EKF::setCovariance(Eigen::MatrixXd P_) {
	P = P_;
}

void EKF::setMatchFlag(int mf) {
	matchFlag = mf;
}

