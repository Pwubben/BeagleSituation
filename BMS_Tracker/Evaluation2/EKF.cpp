
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
	std::cout << x << std::endl;
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

	//std::cout << "Q:\n"<< Q << std::endl;
	R = Eigen::MatrixXd(4, 4); //4 sources of measurement for Beagle
	R << pow(varGPS, 2), 0.0, 0.0, 0.0,
		0.0, pow(varGPS, 2), 0.0, 0.0,
		0.0, 0.0, pow(varYaw, 2), 0.0,
		0.0, 0.0, 0.0, pow(varYawRate, 2);
	//std::cout << "R:\n" << R << std::endl;
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
	init(true),
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

void EKF::compute(Eigen::VectorXd z, double radVel_, double angle_, double omega, Eigen::Vector3d beagleMeas) {
	if (init) {
		init = false;
		lambda = 1;
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
					Z_ << z;// , omega;
				}
				update(Z_, angle_, omega , beagleMeas);
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

	//std::cout << Q << std::endl;
	P = JA * P * JA.transpose() + Q;
	/*if (BeagleObject)
		std::cout << "JA:\n"<< JA << std::endl;*/
}

void EKF::update(const Eigen::VectorXd& z, double angle_, double omega, Eigen::Vector3d beagleMeas) {

	if (init) {
		if (BeagleObject) {
			//Beagle initiation
			x(0) = z(0);
			x(1) = z(1);
			x(2) = z(2);
			x(4) = z(3);
			std::cout << x.transpose() << std::endl;
		}	
	}
	else {
		if (!BeagleObject) {
			if (matchFlag == 1) {
				JH.resize(2, 5);
				JH << 1.0, 0.0, 0.0, 0.0, 0.0,
					0.0, 1.0, 0.0, 0.0, 0.0;
					//0.0, 0.0, 0.0, 0.0, 1.0;
				Hx.resize(2);
				Hx = JH*x;
				//std::cout << "z: " << z.transpose() << std::endl; 
			}
			else if (matchFlag == 0) {
				JH.resize(3, 5);
				JH << 1.0, 0.0, 0.0, 0.0, 0.0,
					  0.0, 1.0, 0.0, 0.0, 0.0,
					  0.0, 0.0, x(3)*sin(M_PI + angle_ - x(2)), cos(M_PI + angle_ - x(2)), 0.0;
				//std::cout << "angle, x(2) x(3) :" << angle_ << ", " << x(2) << ", " << x(3) << std::endl;
				Hx.resize(3);
				Hx << x(0), x(1), x(3)*cos(M_PI + angle_ - x(2));
			}
		}
		/*if (!BeagleObject) {
			if (R.cols() == 1) {
				JH.resize(1, 5);
				JH << (x(1) - beagleMeas(1)) / double(pow(beagleMeas(0), 2) - 2.0*beagleMeas(0)*x(0) + pow(beagleMeas(1), 2) - 2.0*beagleMeas(1)*x(1) + pow(x(0), 2) + pow(x(1), 2)), (beagleMeas(0) - x(0)) / double(pow(beagleMeas(0), 2) - 2.0*beagleMeas(0)*x(0) + pow(beagleMeas(1), 2) - 2.0*beagleMeas(1)*x(1) + pow(x(0), 2) + pow(x(1), 2)), 0.0, 0.0, 0.0;
				Hx.resize(1);
				Hx << Util::constrainAngle(atan2((x(0) - beagleMeas(0)), (x(1) - beagleMeas(1))) - beagleMeas(2));
			}
			else {
				JH.resize(3, 5);
				JH << (x(0) - beagleMeas(0)) / double(sqrt(pow(x(0) - beagleMeas(0), 2) + pow(x(1) - beagleMeas(1), 2))), (x(1) - beagleMeas(1)) / double(sqrt(pow(x(0) - beagleMeas(0), 2) + pow(x(1) - beagleMeas(1), 2))), 0.0, 0.0, 0.0,
					(x(1)-beagleMeas(1))/double(pow(beagleMeas(0),2)-2.0*beagleMeas(0)*x(0)+pow(beagleMeas(1),2)-2.0*beagleMeas(1)*x(1)+pow(x(0),2)+pow(x(1),2)) , (beagleMeas(0)- x(0)) / double(pow(beagleMeas(0), 2) - 2.0*beagleMeas(0)*x(0) + pow(beagleMeas(1), 2) - 2.0*beagleMeas(1)*x(1) + pow(x(0), 2) + pow(x(1), 2)), 0.0, 0.0, 0.0,
					0.0, 0.0, x(3)*sin(M_PI + angle_ - x(2)), cos(M_PI + angle_ - x(2)), 0.0;
				Hx.resize(3);
				Hx << sqrt(pow(x(0) - beagleMeas(0), 2) + pow(x(1) - beagleMeas(1), 2)), Util::constrainAngle(atan2((x(0) - beagleMeas(0)),( x(1) - beagleMeas(1))) - beagleMeas(2)), x(3)*cos(M_PI + angle_ - x(2));
			}
		}*/
		
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
			modelProbability(P, R, z);

		if (!BeagleObject) {
			//std::cout << "zm: \n" << z.transpose() << std::endl;
			//std::cout << "Z: \n" << Z.transpose() << std::endl;
			//std::cout << "R: \n" << R << std::endl;
			//std::cout << "P:\n" << P << std::endl;
			//std::cout << "S1:\n" << S.inverse() << std::endl;
			//std::cout << "K: \n" << K << std::endl;
			//std::cout << "x b: \n" << x.transpose() << std::endl;
			//std::cout << "JH1: \n" << JH << std::endl;
			//std::cout << "Hx: \n" << Hx << std::endl;
		}
		// Update estimate
		x = x + K * (Z);

		if (BeagleObject) {
			//std::cout << "x_beagle\n " << x.transpose() << std::endl;
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
			//std::cout << "x_target:\n" << x.transpose() << std::endl;
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
			JA << 1.0, 0.0, x(3)*dt*cos(x(2)), dt*sin(x(2)), 0.0,
				0.0, 1.0, -x(3)*dt*sin(x(2)), dt*cos(x(2)), 0.0,
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
		//std::cout << "JA:: \n" << JA << std::endl;
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
		double omega = Util::deg2Rad(-10);
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
		double omega = Util::deg2Rad(10);
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

	//Ship model
	if (modelNum == 9) {
		double L = 25;
		double V = 56;
		double D = 0.0222 / sqrt(9.81*pow(V, 1.0 / 3.0))*pow(L / pow(V, 1.0 / 3.0), 2.85);
		double FroudeNr = x(3) / sqrt(9.81*pow(V, 1.0 / 3.0));
		double DepthNr = L / pow(V, 1.0 / 3.0);
		//std::cout << "Froude Number: " << FroudeNr << " - Depth: " << DepthNr << std::endl;
		int n = x.size();
		JA = Eigen::MatrixXd(n, n);

		if (fabs(x(4)) < 0.005) {
			x(0) = x(0) + (x(3) * dt) * sin(x(2));
			x(1) = x(1) + (x(3) * dt) * cos(x(2));
			x(2) = std::fmod((x(2) + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = 45;// Util::sgn(x(4))*std::max(double(abs(x(4))), double(0.0001));
			//std::cout << "x: \n" << x << std::endl;
			//std::cout << "xq: \n" << (L*(pow(D*x(3) + 1.7, 2)), dt*x(3) / (15.0*L*(D*(x(3) + 1.7)))) << std::endl;
			if (fabs(x(4)) < 0.01) {
				JA << 1.0, 0.0, x(3)*dt*cos(x(2)), dt*sin(x(2)), 0.0,
					0.0, 1.0, -x(3)*dt*sin(x(2)), dt*cos(x(2)), 0.0,
					0.0, 0.0, 1.0, (3.4 / 30.0) / (L*(pow(D*x(3) + 1.7, 2))), dt*x(3) / (15.0*L*(D*(x(3) + 1.7))),
					0.0, 0.0, 0.0, 1.0, 0.0,
					0.0, 0.0, 0.0, 0.0, 1.0;
			}
		}
		else {
			x(0) = x(0) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*x(4))) * (-cos(x(3)*dt / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4))))  + x(2)) + cos(x(2)));
			x(1) = x(1) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*x(4))) * (sin(x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) * dt + x(2)) - sin(x(2)));
			x(2) = std::fmod((x(2) + x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) * dt + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = 45;//Util::sgn(x(4))*std::min(double(abs(x(4))), double(60));
			//std::cout << "omega: " << x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) << std::endl;
			//std::cout << "d2: " << (L*(1.7 + D*x(3)))*(30 / double(2.0*x(4))) << std::endl;
			
			//std::cout << "x: \n" << x << std::endl;
			JA = Eigen::MatrixXd(x.size(), x.size());
			JA << 1.0, 0.0, -5.51819*L*(D*x(3) + 1.7)*(sin(x(2)) - sin((0.181219 *dt * x(3)) / (D*L*x(3) + 1.7*L) + x(2))), 15.0*D*L*(cos(x(2)) - cos((x(3)*dt*x(4)) / double(15.0*L*(D*x(3) + 1.7)) + x(2))) / x(4) + 15.0*L*(D*x(3) + 1.7) / x(4)*((x(4)*dt) / double(15.0*L*(D*x(3) + 1.7)) - x(4)*D*x(3)*dt / double(15.0*L*pow((D*x(3) + 1.7), 2)))*sin(x(4)*x(3)*dt / double(15.0*L*(D*x(3) + 1.7)) + x(2)), dt*x(3) / x(4)*sin(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - 15.0*L*(D*x(3) + 1.7)*(-cos(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) + cos(x(2))) / pow(x(4), 2),
				0.0, 1.0, -5.51819*L*(D*x(3) + 1.7)*(cos(x(2)) - cos((0.181219 *dt * x(3)) / (D*L*x(3) + 1.7*L) + x(2))), 15.0*D*L*(-sin(x(2)) + sin((x(3)*dt*x(4)) / double(15.0 * L*(D*x(3) + 1.7)) + x(2))) / x(4) + 15.0*L*(D*x(3) + 1.7) / x(4)*((x(4)*dt) / double(15.0*L*(D*x(3) + 1.7)) - x(4)*D*x(3)*dt / double(15.0*L*pow((D*x(3) + 1.7), 2)))*cos(x(4)*x(3)*dt / double(15.0*L*(D*x(3) + 1.7)) + x(2)), dt*x(3) / x(4)*cos(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - 15.0*L*(D*x(3) + 1.7)*(sin(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - sin(x(2))) / pow(x(4), 2),
				0.0, 0.0, 1.0, 25.5*dt/(x(4)*L*pow(D*x(3)+1.7,2)), 15.0*dt*x(3) / (pow(x(4),2)*L*(D*(x(3) + 1.7))),
				0.0, 0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 1.0;
			
		}
		//std::cout << "JA:\n" << JA << std::endl;
	}
	if (modelNum == 10) {
		double omega = 45.0;
		double L = 25;
		double V = 56;
		double D = 0.0222 / sqrt(9.81*pow(V, 1.0 / 3.0))*pow(L / pow(V, 1.0 / 3.0), 2.85);
		double FroudeNr = x(3) / sqrt(9.81*pow(V, 1.0 / 3.0));
		double DepthNr = L / pow(V, 1.0 / 3.0);
		//std::cout << "Froude Number: " << FroudeNr << " - Depth: " << DepthNr << std::endl;
		int n = x.size();
		JA = Eigen::MatrixXd(n, n);

			x(0) = x(0) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*omega)) * (-cos(x(3)*dt / ((L*(1.7 + D*x(3)))*(30 / double(2.0*omega))) + x(2)) + cos(x(2)));
			x(1) = x(1) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*omega)) * (sin(x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*omega))) * dt + x(2)) - sin(x(2)));
			x(2) = std::fmod((x(2) + x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) * dt + M_PI), (2.0 * M_PI)) - M_PI;
			x(3) = abs(x(3));
			x(4) = omega;// Util::sgn(x(4))*std::min(double(abs(x(4))), double(60));
			//std::cout << "omega: " << x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) << std::endl;
			//std::cout << "d2: " << (L*(1.7 + D*x(3)))*(30 / double(2.0*x(4))) << std::endl;

			//std::cout << "x: \n" << x.transpose() << std::endl;
			JA = Eigen::MatrixXd(x.size(), x.size());
			JA << 1.0, 0.0, -5.51819*L*(D*x(3) + 1.7)*(sin(x(2)) - sin((0.181219 *dt * x(3)) / (D*L*x(3) + 1.7*L) + x(2))), 15.0*D*L*(cos(x(2)) - cos((x(3)*dt*x(4)) / double(15.0*L*(D*x(3) + 1.7)) + x(2))) / x(4) + 15.0*L*(D*x(3) + 1.7) / x(4)*((x(4)*dt) / double(15.0*L*(D*x(3) + 1.7)) - x(4)*D*x(3)*dt / double(15.0*L*pow((D*x(3) + 1.7), 2)))*sin(x(4)*x(3)*dt / double(15.0*L*(D*x(3) + 1.7)) + x(2)), dt*x(3) / x(4)*sin(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - 15.0*L*(D*x(3) + 1.7)*(-cos(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) + cos(x(2))) / pow(x(4), 2),
				0.0, 1.0, -5.51819*L*(D*x(3) + 1.7)*(cos(x(2)) - cos((0.181219 *dt * x(3)) / (D*L*x(3) + 1.7*L) + x(2))), 15.0*D*L*(-sin(x(2)) + sin((x(3)*dt*x(4)) / double(15.0 * L*(D*x(3) + 1.7)) + x(2))) / x(4) + 15.0*L*(D*x(3) + 1.7) / x(4)*((x(4)*dt) / double(15.0*L*(D*x(3) + 1.7)) - x(4)*D*x(3)*dt / double(15.0*L*pow((D*x(3) + 1.7), 2)))*cos(x(4)*x(3)*dt / double(15.0*L*(D*x(3) + 1.7)) + x(2)), dt*x(3) / x(4)*cos(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - 15.0*L*(D*x(3) + 1.7)*(sin(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - sin(x(2))) / pow(x(4), 2),
				0.0, 0.0, 1.0, 25.5*dt / (x(4)*L*pow(D*x(3) + 1.7, 2)), 15.0*dt*x(3) / (pow(x(4), 2)*L*(D*(x(3) + 1.7))),
				0.0, 0.0, 0.0, 1.0, 0.0,
				0.0, 0.0, 0.0, 0.0, 0.0;
	}
	if (modelNum == 11) {
		double omega = -45.0;
		double L = 25;
		double V = 56;
		double D = 0.0222 / sqrt(9.81*pow(V, 1.0 / 3.0))*pow(L / pow(V, 1.0 / 3.0), 2.85);
		double FroudeNr = x(3) / sqrt(9.81*pow(V, 1.0 / 3.0));
		double DepthNr = L / pow(V, 1.0 / 3.0);
		//std::cout << "Froude Number: " << FroudeNr << " - Depth: " << DepthNr << std::endl;
		int n = x.size();
		JA = Eigen::MatrixXd(n, n);

		x(0) = x(0) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*omega)) * (-cos(x(3)*dt / ((L*(1.7 + D*x(3)))*(30 / double(2.0*omega))) + x(2)) + cos(x(2)));
		x(1) = x(1) + (L*(1.7 + D*x(3)))*(30.0 / double(2.0*omega)) * (sin(x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*omega))) * dt + x(2)) - sin(x(2)));
		x(2) = std::fmod((x(2) + x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) * dt + M_PI), (2.0 * M_PI)) - M_PI;
		x(3) = abs(x(3));
		x(4) = omega;// Util::sgn(x(4))*std::min(double(abs(x(4))), double(60));
		//std::cout << "omega: " << x(3) / ((L*(1.7 + D*x(3)))*(30 / double(2.0*x(4)))) << std::endl;
		//std::cout << "d2: " << (L*(1.7 + D*x(3)))*(30 / double(2.0*x(4))) << std::endl;

		//std::cout << "x: \n" << x.transpose() << std::endl;
		JA = Eigen::MatrixXd(x.size(), x.size());
		JA << 1.0, 0.0, -5.51819*L*(D*x(3) + 1.7)*(sin(x(2)) - sin((0.181219 *dt * x(3)) / (D*L*x(3) + 1.7*L) + x(2))), 15.0*D*L*(cos(x(2)) - cos((x(3)*dt*x(4)) / double(15.0*L*(D*x(3) + 1.7)) + x(2))) / x(4) + 15.0*L*(D*x(3) + 1.7) / x(4)*((x(4)*dt) / double(15.0*L*(D*x(3) + 1.7)) - x(4)*D*x(3)*dt / double(15.0*L*pow((D*x(3) + 1.7), 2)))*sin(x(4)*x(3)*dt / double(15.0*L*(D*x(3) + 1.7)) + x(2)), dt*x(3) / x(4)*sin(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - 15.0*L*(D*x(3) + 1.7)*(-cos(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) + cos(x(2))) / pow(x(4), 2),
			0.0, 1.0, -5.51819*L*(D*x(3) + 1.7)*(cos(x(2)) - cos((0.181219 *dt * x(3)) / (D*L*x(3) + 1.7*L) + x(2))), 15.0*D*L*(-sin(x(2)) + sin((x(3)*dt*x(4)) / double(15.0 * L*(D*x(3) + 1.7)) + x(2))) / x(4) + 15.0*L*(D*x(3) + 1.7) / x(4)*((x(4)*dt) / double(15.0*L*(D*x(3) + 1.7)) - x(4)*D*x(3)*dt / double(15.0*L*pow((D*x(3) + 1.7), 2)))*cos(x(4)*x(3)*dt / double(15.0*L*(D*x(3) + 1.7)) + x(2)), dt*x(3) / x(4)*cos(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - 15.0*L*(D*x(3) + 1.7)*(sin(x(4)*dt*x(3) / double(15.0*L*(D*x(3) + 1.7)) + x(2)) - sin(x(2))) / pow(x(4), 2),
			0.0, 0.0, 1.0, 25.5*dt / (x(4)*L*pow(D*x(3) + 1.7, 2)), 15.0*dt*x(3) / (pow(x(4), 2)*L*(D*(x(3) + 1.7))),
			0.0, 0.0, 0.0, 1.0, 0.0,
			0.0, 0.0, 0.0, 0.0, 0.0;
	}
}

void EKF::modelProbability(Eigen::MatrixXd P, Eigen::MatrixXd R, const Eigen::VectorXd& z) {
	Eigen::MatrixXd JHProb(2, 5);
	JHProb << 1.0, 0.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0, 0.0;

	Eigen::MatrixXd JHT = P * JH.transpose();
	Eigen::MatrixXd S = JH * JHT + R;
	//if (matchFlag == 0) {
	//std::cout << "S:\n" << S << std::endl;
	//	std::cout << "P:\n" << P << std::endl;
	//}
	lambda = 1.0 / (2.0 * M_PI * sqrt(S.determinant())) * std::exp(-0.5*(z-Hx).transpose()*S.inverse()*(z - Hx));

	lambda *= 100000;
	//std::cout << "Hx:\n" << Hx << std::endl;
	//std::cout << "z:\n" << z << std::endl;
	//std::cout << "l: " << lambda << std::endl;
	if (lambda < 1e-60)
		lambda = 1e-60;

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

