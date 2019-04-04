#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

Track::Track(double range, double angle, const double& velocity, Eigen::Vector4d beagleMeas, int objectChoice_, evaluationSettings settings) :
	detectionAbsence(0),
	objectChoice(objectChoice_),
	matchFlag_(0),
	evalSettings(settings)
{
	range_ = range;
	angle_ = angle;
	radVel_ = velocity - cos(angle_)*beagleMeas(3);

	body2nav(range, angle, beagleMeas);

	x_measVec.push_back(navDet(0));
	y_measVec.push_back(navDet(1));

	//Measurement variables
	g = Eigen::MatrixXd(navDet.size(), navDet.size());

	//IMM parameters
	modelNumbers = settings.dynamicModels;
	modelNum = modelNumbers.size();
	Eigen::MatrixXd stateTransitionProb(modelNum, modelNum);
	if (modelNum == 4) {
		stateTransitionProb << 0.91, 0.03, 0.03, 0.03,
			0.03, 0.91, 0.03, 0.03,
			0.03, 0.03, 0.91, 0.03,
			0.03, 0.03, 0.03, 0.91;
	}
	if (modelNum == 3) {
		stateTransitionProb << 0.92, 0.04, 0.04,
			0.04, 0.92, 0.03,
			0.04, 0.04, 0.92;
	}
	if (modelNum == 2) {
		stateTransitionProb << 0.96, 0.04,
			0.04, 0.96;
			
	}
	if (modelNum == 1) {
		stateTransitionProb << 1;
	}
	//Result Vector
	Eigen::VectorXd radMeasurement(6);
	radMeasurement << count, range_, angle_, radVel_, navDet;
	radarMeasurement.push_back(radMeasurement);

	//Set initial matrices
	tuning();

	double headingInit = atan2(beagleMeas(0) - navDet(0), beagleMeas(1) - navDet(1));
	if (radVel_ < 0)
		headingInit += M_PI;

	//Initial velocity estimates [m/s]
	double vInit;
	if (velocity == -100 || objectChoice == 0)
		vInit = 6.0;
	else
		vInit = abs(radVel_);

	//Kalman filter track
	if (objectChoice == 0)
		KF = std::make_unique<Kalman>(navDet, vInit, headingInit, Qvec[0], Pvec[0]);
	else if (objectChoice == 1)
		EKF_ = std::make_unique<EKF>(navDet, vInit, headingInit, Qvec[1], Pvec[1], settings.dynamicModels[0], beagleMeas.head(3));
	else {
		IMM_ = std::make_unique<IMM>(modelNum, modelNumbers, Qvec, Pvec, navDet, vInit, headingInit, beagleMeas.head(3));
		IMM_->setStateTransitionProbability(stateTransitionProb);
	}
}

void Track::tuning() {
	//Move function to header


	//Model 0 - Constant velocity
	tuningVec.push_back(Tuning({
		0.0001, 
		0.00005,      //Radar measurement noise covariance
		0.0001,	  //rangeVarCamera
		0.005,      //angleVarCamera
		1.0       //varianceTimeFactor	
	}));
	Pvec.push_back(Eigen::MatrixXd(4, 4));
	Qvec.push_back(Eigen::MatrixXd(4, 4));
	Pvec.back() << 2.0, 0, 0, 0,
		0, 4.0, 0, 0,
		0, 0, 2.0, 0,
		0, 0, 0, 4.0;

	Qvec.back() << 3.0, 0, 0, 0,
		0,1.0, 0, 0,
		0, 0, 3.0, 0,
		0, 0, 0, 1.0;

	////Model 1 & 2 - Constant turn 30deg/s
	//tuningVec.insert(tuningVec.end(), 2, Tuning({
	//	0.001,
	//	0.0001,      //Radar measurement noise covariance
	//	0.1,	  //rangeVarCamera
	//	0.1,      //angleVarCamera
	//	1.05       //varianceTimeFactor	
	//}));
	//Pvec.push_back(Eigen::MatrixXd(4, 4));
	//Qvec.push_back(Eigen::MatrixXd(4, 4));
	//Pvec.back() << 10, 0, 0, 0,
	//	0, 2, 0, 0,
	//	0, 0, 10, 0,
	//	0, 0, 0, 2;

	//Qvec.back() << 5, 0, 0, 0,
	//	0, 2, 0, 0,
	//	0, 0, 5, 0,
	//	0, 0, 0, 2;
	//Pvec.push_back(Pvec.back());
	//Qvec.push_back(Qvec.back());

	////Model 3 & 4 - Constant turn 50deg/s
	//tuningVec.insert(tuningVec.end(), 2, Tuning({
	//	0.001,
	//	0.0001,      //Radar measurement noise covariance
	//	0.1,	  //rangeVarCamera
	//	0.1,      //angleVarCamera
	//	1.05       //varianceTimeFactor	
	//}));
	//Pvec.push_back(Eigen::MatrixXd(4, 4));
	//Qvec.push_back(Eigen::MatrixXd(4, 4));
	//Pvec.back() << 10, 0, 0, 0,
	//	0, 2, 0, 0,
	//	0, 0, 10, 0,
	//	0, 0, 0, 2;

	//Qvec.back() << 5, 0, 0, 0,
	//	0, 2, 0, 0,
	//	0, 0, 5, 0,
	//	0, 0, 0, 2;
	//Pvec.push_back(Pvec.back());
	//Qvec.push_back(Qvec.back());

	//Model 5 - EKF - Constant Velocity Constant Turn

	int seastate = 1;

	if (seastate == 1) {
		tuningVec.push_back(Tuning({
			0.0001,
			0.00005,      //Radar measurement noise covariance Was 0.00005
			0.0001,	  //rangeVarCamera
			0.005,      //angleVarCamera
			1       //varianceTimeFactor	
		}));
		//tuningVec.push_back(Tuning({
		//	0.001,
		//	0.00001,      //Radar measurement noise covariance
		//	0.001,	  //rangeVarCamera
		//	0.05,      //angleVarCamera
		//	1.0       //varianceTimeFactor	
		//}));

		if (evalSettings.varianceFactor != 0)
			tuningVec.back().varianceTimeFactor = evalSettings.varianceFactor;
		Pvec.push_back(Eigen::MatrixXd(5, 5));
		Qvec.push_back(Eigen::MatrixXd(5, 5));
		Pvec.back() <<
			3.0, 0, 0, 0, 0,
			0, 3.0, 0, 0, 0,
			0, 0, 1.0, 0, 0,
			0, 0, 0, 4.0, 0,
			0, 0, 0, 0, 0.001;
		Qvec.back() <<
			2.0, 0, 0, 0, 0,
			0, 2.0, 0, 0, 0,
			0, 0, 0.05, 0, 0,
			0, 0, 0, 1.5, 0,
			0, 0, 0, 0, 0.001;
		/*Qvec.back() <<
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0.2, 0,
			0, 0, 0, 0, 0;*/
		//Qvec.back() = Eigen::MatrixXd::Zero(5, 5);
		std::cout << Qvec[1] << std::endl;
	}
	if (seastate == 3) {
		tuningVec.push_back(Tuning({
			0.001,
			0.0001,      //Radar measurement noise covariance
			0.001,	  //rangeVarCamera
			0.005,      //angleVarCamera
			1       //varianceTimeFactor	
		}));
		//tuningVec.push_back(Tuning({
		//	0.001,
		//	0.00001,      //Radar measurement noise covariance
		//	0.001,	  //rangeVarCamera
		//	0.05,      //angleVarCamera
		//	1.0       //varianceTimeFactor	
		//}));

		if (evalSettings.varianceFactor != 0)
			tuningVec.back().varianceTimeFactor = evalSettings.varianceFactor;
		Pvec.push_back(Eigen::MatrixXd(5, 5));
		Qvec.push_back(Eigen::MatrixXd(5, 5));
		Pvec.back() <<
			3, 0, 0, 0, 0,
			0, 3, 0, 0, 0,
			0, 0, 1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 1;
		Qvec.back() <<
			4, 0, 0, 0, 0,
			0, 4, 0, 0, 0,
			0, 0, 0.1, 0, 0,
			0, 0, 0, 1, 0,
			0, 0, 0, 0, 5;
	}
}

void Track::run(Eigen::Vector3d _beaglePrediction) {
	
	updateR(_beaglePrediction);

	if (objectChoice == 0) {
		KF->setMatchFlag(matchFlag_);
		KF->compute(navDet, radVel_, angle_+_beaglePrediction(2), omega_);
		stateVector.push_back(KF->getState());
	}
	else if (objectChoice == 1) {
		EKF_->setMatchFlag(matchFlag_);
		EKF_->compute(navDet,radVel_, angle_ + _beaglePrediction(2), omega_,_beaglePrediction);
		stateVector.push_back(EKF_->getState());
	}
	else {
		IMM_->setMatchFlag(matchFlag_);
		IMM_->run(navDet,radVel_, angle_ + _beaglePrediction(2), omega_, _beaglePrediction);
		stateVector.push_back(IMM_->getState());
		muVector.push_back(IMM_->getMu());
		/*std::cout <<"navDet: \n"<< navDet << std::endl;*/
	}
	
	
	//Compute detection prediction by combining estimates from Beagle and Track
	nav2body(_beaglePrediction);

	count++;
}

void Track::updateR(Eigen::Vector3d _beaglePrediction) {
	
	g << sin(_beaglePrediction(2) + angle_), range_*cos(_beaglePrediction(2) + angle_),
	cos(_beaglePrediction(2) + angle_), -range_*sin(_beaglePrediction(2) + angle_);

	if (objectChoice == 0) {
		Eigen::Matrix2d Rc, Rr, R;
		if (matchFlag_ == 0) {
			Rr << tuningVec[0].rangeVarRadar, 0,
				0, tuningVec[0].angleVarRadar;
			R = g*Rr*g.transpose();
		}
		else if (matchFlag_ == 1) {
			Rc << tuningVec[0].rangeVarCamera *pow(tuningVec[0].varianceTimeFactor, detectionAbsence), 0,
				0, tuningVec[0].angleVarCamera;
			R = g*Rc*g.transpose();
		}
		KF->setR(R);
	}
	else if (objectChoice == 1) {
		Eigen::Matrix2d Rc, Rr, R;
		Eigen::Matrix3d RrVel;
		RrVel = Eigen::Matrix3d::Zero(3, 3);
		if (matchFlag_ == 0) {
			Rr << tuningVec[1].rangeVarRadar, 0,
				0, tuningVec[1].angleVarRadar;
			R = g*Rr*g.transpose();
			
			if (radVel_ != -100) {
				RrVel.block(0, 0, 2, 2) = R;
				RrVel(2, 2) = 0.001;
			}
		}
		else if (matchFlag_ == 1) {
			Rc << tuningVec[1].rangeVarCamera *pow(tuningVec[1].varianceTimeFactor, detectionAbsence), 0,
				0, tuningVec[1].angleVarCamera;
			R = g*Rc*g.transpose();
			RrVel.block(0, 0, 2, 2) = R;
			RrVel(2, 2) = 0.00001;
		}
		if (matchFlag_ == 1 || radVel_ == -100)
			EKF_->setR(R);
		else
			EKF_->setR(RrVel);
		
	}
	else {
		std::vector<Eigen::MatrixXd> Rvec;
		for (auto &&i : modelNumbers) {
			//Eigen::MatrixXd Rc(1,1);
			Eigen::Matrix2d Rc, Rr, R;
			Eigen::Matrix3d RrVel;
			RrVel = Eigen::Matrix3d::Zero(3,3);
			if (matchFlag_ == 0) {
	
				Rr << tuningVec[i>=5].rangeVarRadar, 0,
						0, tuningVec[i>=5].angleVarRadar;
				R = g*Rr*g.transpose();
				
				if (i >= 5) {
					RrVel.block(0, 0, 2, 2) = R;
					RrVel(2, 2) = 0.001;
				}
				
				/*RrVel << tuningVec[i>5].rangeVarRadar, 0, 0,
					0, tuningVec[i>5].angleVarRadar,0,
					0, 0, 0.01;*/
			}
			else if (matchFlag_ == 1) {
				Rc << tuningVec[i>=5].rangeVarCamera *pow(tuningVec[i>=5].varianceTimeFactor, detectionAbsence), 0,
					0, tuningVec[i>=5].angleVarCamera;
				R = g*Rc*g.transpose();
				RrVel.block(0, 0, 2, 2) = R;
				RrVel(2, 2) = 0.00001;
				//Rc << tuningVec[i > 5].angleVarCamera;
			}
			if (i < 5 || matchFlag_ == 1 || radVel_ == -100)
				Rvec.push_back(R);
			else 
				Rvec.push_back(RrVel);
		}
		IMM_->setR(Rvec);
		//std::cout << "Rvec:\n" << Rvec[0] << "\n" << Rvec[1] << std::endl;
	}
}

prediction Track::getPrediction() {
	return prediction_;
}

double Track::getDetection() {
	return range_;
}

double Track::getAngle() {
	return angle_;
}

void Track::setDetection(const double& range,const double& angle, const double& velocity, const double& omega, Eigen::Vector4d beagleMeas, int matchFlag, double boundRectx) {
	
	matchFlag_ = matchFlag;

	if (matchFlag == 0) {
		//std::cout << "\n" << std::endl;
		range_ = range;
		angle_ = angle;
		radVel_ = velocity - cos(angle_)*beagleMeas(3);
		std::cout << "beaglemeas \n" << beagleMeas(3) << std::endl;
		std::cout << "radvel \n" << radVel_ << std::endl;
	}
	if (matchFlag_ == 1)
		omega_ = omega;
	//Compute detection in navigation frame coordinates
	body2nav(range, angle, beagleMeas); //TODO Body2Nav - Perhaps we need predictions later instead of measurements

	//if (navDet(1) > 400)
	//	//cv::waitKey(0);


	if (matchFlag_ != 2) {

		//std::cout << "MatchFlag: " << matchFlag_ << "- Range: " << range << "- Angle: " << angle << "navdet:" << navDet << std::endl;

		if (matchFlag == 0) {
			Eigen::VectorXd radMeasurement(6);
			radMeasurement << count, range_, angle_, radVel_, navDet;
			radarMeasurement.push_back(radMeasurement);
		}
		if (matchFlag == 1) {
			Eigen::VectorXd camMeasurement(6);
			camMeasurement << count ,range, angle, navDet, boundRectx;
			cameraMeasurement.push_back(camMeasurement);
		}

 		//std::cout << "BeagleMeas " << beagleMeas(0) << " - " << beagleMeas(1) << " - " << beagleMeas(2) << std::endl;
		x_measVec.push_back(navDet(0));
		y_measVec.push_back(navDet(1));

		//std::cout << "x: " << navDet(0) << " - y: " << navDet(1) << "\n" << std::endl;
	}
}

void Track::body2nav(const double& range, const double& angle, Eigen::Vector4d& beagleMeas) {
	
	relDet << sin(angle+ beagleMeas(2))*range, cos(angle + beagleMeas(2))*range;

	//Compute detection in navigation frame coordinates

	//std::cout << "Meas: " << beagleMeas.transpose() << std::endl;
	navDet = relDet + beagleMeas.head(2);
	//if (matchFlag_ == 0) {
	//	navEkf.resize(2);
	//	navEkf << range, angle;
	//}
	//if (matchFlag_ == 1) {
	//	navEkf.resize(1);
	//	navEkf << angle;
	//}
}

void Track::nav2body(Eigen::Vector3d _beaglePrediction) {

	Eigen::Vector2d z_predict;
	
	if (objectChoice == 0)
		z_predict = KF->getPrediction();
	else if (objectChoice == 1)
		z_predict = EKF_->getPrediction();
	else
		z_predict = IMM_->getPrediction();


	//std::cout << "Z_predict: " << z_predict << std::endl;

	//x y prediction target relative to beagle position prediction
	Eigen::Vector2d pdTarget = z_predict - _beaglePrediction.head(2);

	//Create plot vectors containing predictions
	x_predictVec.push_back(z_predict(0));
	y_predictVec.push_back(z_predict(1));

	//Rotate to body frame using beagle heading prediction
	/*rotMat << cos(_beaglePrediction(2)), -sin(_beaglePrediction(2)),
		sin(_beaglePrediction(2)), cos(_beaglePrediction(2));*/

	//Compute detection in body frame coordinates
	prediction_coord = rotMat*pdTarget;

	//Compute range and angle
	prediction_.range = sqrt(pow(pdTarget(0), 2) + pow(pdTarget(1), 2));
	prediction_.angle = atan2(pdTarget(0), pdTarget(1)) - _beaglePrediction(2);

	//std::cout << "Beaglepred: " << _beaglePrediction.transpose() << std::endl;
	//std::cout <<"angle: "<< prediction_.angle << std::endl;
}

std::vector<std::vector<double>> Track::getPlotVectors() {
	std::vector<std::vector<double>> plotVectors;
	plotVectors.push_back(x_measVec);
	plotVectors.push_back(y_measVec);
	plotVectors.push_back(x_predictVec);
	plotVectors.push_back(y_predictVec);
	return plotVectors;
}

std::vector<std::vector<Eigen::VectorXd>> Track::getResultVectors() {
	std::vector<std::vector<Eigen::VectorXd>> resultVectors;
	resultVectors.push_back(radarMeasurement);
	resultVectors.push_back(cameraMeasurement);
	resultVectors.push_back(stateVector);
	resultVectors.push_back(muVector);
	
	return resultVectors;
}