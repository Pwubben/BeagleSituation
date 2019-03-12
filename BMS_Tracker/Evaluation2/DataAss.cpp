#include "stdafx.h"
#include "GnuGraph.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace cv;

GnuGraph graph;

void DataAss::run(const detection& info) {

	//Update Beagle state
	BeagleTrack->update(_beagleMeas);
	_beaglePrediction = BeagleTrack->getBeaglePrediction();

	//Retrieve predictions and detections from last step
	std::vector<prediction> predictionVector(tracks_.size());
	std::vector<double> lastDetection(tracks_.size());
	std::vector<double> lastAngle(tracks_.size());
	vector<bool> newRangeDetection(info.radarRange.size(), true);
	vector<bool> newAngleDetection(info.radarAngle.size(), true);

	//std::cout << "Tracks: " << tracks_.size() << std::endl;

	//Empty matched detection vector
	detect.relRange.clear();
	detect.relAngle.clear();
	detect.relVel.clear();

	for (int i = 0; i < tracks_.size(); i++) {//Change later to more robust form - PDA
		predictionVector[i] = tracks_[i].getPrediction();
		lastDetection[i] = tracks_[i].getDetection();
		lastAngle[i] = tracks_[i].getAngle();
		//Check for range update
		std::pair<bool,int> result = findInVector(info.radarRange, lastDetection[i]);
		std::pair<bool,int> resultAngle = findInVector(info.radarAngle, lastAngle [i]);
		//std::pair<bool, int> result = findRangeVector(info.radarRange, lastDetection[i], 2.0);

		if (result.first) 
			newRangeDetection[result.second] = false;
		if (resultAngle.first)
			newAngleDetection[resultAngle.second] = false;
	}

	//Match radar and camera detections in case of new radar detection
	for (int i = 0; i < newRangeDetection.size(); i++) {
		if (newRangeDetection[i] || newAngleDetection[i]) {

			detect.relRange.push_back(info.radarRange[i]);
			detect.relAngle.push_back(info.radarAngle[i]);
			detect.relVel.push_back(info.radarVel[i]);
		}
	}

	vector<bool> unassignedDetection(detect.relAngle.size(), true);
	vector<int> matchFlag(tracks_.size(), -1);

	radarCount++;

	//Match detections to tracks
	GlobalNearestNeighbor(info, predictionVector, unassignedDetection, lastDetection, matchFlag);

	//For ground truth velocity
	TargetTrack->update(_targetMeas);

	BeagleState.push_back(BeagleTrack->getState());
	TargetState.push_back(TargetTrack->getState());

	TargetTrack->compute(_targetMeas);

	//TODO Beagle KF - Obtain Beagle updates 
	BeagleTrack->compute(_beagleMeas);
	_beaglePrediction = BeagleTrack->getBeaglePrediction(); //This prediction is correct for the track runs, for the previously used ones might have to use updated state

															//Initiate track if detection is not assigned
	for (int i = 0; i < unassignedDetection.size(); i++) {
		if (unassignedDetection[i] && tracks_.size() < 10) {
			std::cout << "Range: " << detect.relRange[i] << "- Angle: " << detect.relAngle[i] << std::endl;
			tracks_.push_back(Track(detect.relRange[i], detect.relAngle[i], detect.relVel[i], _beaglePrediction, objectChoice, evalSettings));
			matchFlag.push_back(0);
		}
	}

	//Run each track for new predictions
	for (int i = 0; i < tracks_.size(); i++) {
		tracks_[i].run(_beaglePrediction.head(3));
	}

	drawCount++;
	//Draw results
	if (drawCount > 1700) {
		drawResults();
		drawCount = 0;
	}
}

void DataAss::GlobalNearestNeighbor(const detection& info, std::vector<prediction> predictionVector, std::vector<bool>& unassignedDetection, std::vector<double> lastDetection, std::vector<int> matchFlag) {
	
	//Global nearest match solution
	vector<int> trackMatch;
	vector<int> camMatch;
	angleMatchThres = Util::deg2Rad(2);

		if (!detect.relRange.empty() && tracks_.size() > 0)
			trackMatch = distancePL(detect, predictionVector);
		if (!info.cameraAngle.empty() && tracks_.size() > 0)
			camMatch = distanceDet(info, predictionVector);

		// Match radar detections if match is found
		for (int i = 0; i < tracks_.size(); i++) {
			if (!detect.relRange.empty() && trackMatch[i] != -1) {
				matchFlag[i] = 0;
				tracks_[i].detectionAbsence = 0;
				tracks_[i].setDetection(detect.relRange[trackMatch[i]], detect.relAngle[trackMatch[i]], detect.relVel[trackMatch[i]], _beaglePrediction, matchFlag[i]);
				unassignedDetection[trackMatch[i]] = false;
			}

			//Match camera detection if no radar update or no gated match
			if (matchFlag[i] == -1 && !info.cameraAngle.empty() && camMatch[i] != -1 && cameraUtil) {
					//Match flag for kalman update	
					matchFlag[i] = 1;
					tracks_[i].detectionAbsence++;

					//Range prediction is returned as detection
					tracks_[i].setDetection(predictionVector[i].range, info.cameraAngle[camMatch[i]], -100, _beaglePrediction, matchFlag[i]);
			}

			//Return prediction as measurement if no match is found
			if (matchFlag[i] == -1) {
				matchFlag[i] = 2;
				tracks_[i].detectionAbsence++;//TODO detectionAbsence - link dt 
				tracks_[i].setDetection(predictionVector[i].range, predictionVector[i].angle, -100, _beaglePrediction, matchFlag[i]);

			}

			//Terminate track if no radar detection has been received for too long
			if (tracks_[i].detectionAbsence > absenceThreshold)
				tracks_.erase(tracks_.begin() + i); //Check if correct one is erased

		}
}

void DataAss::setBeagleData(Eigen::Vector4d& beagleData_) {



	//Set initial Beagle position as reference point - if first
	if (beagleInit) {
		//Convert lon-lat to x-y
		aspectRatio = cos(beagleData_(0)/double(60));
		xyInit << earthRadius* Util::deg2Rad(beagleData_(0)) / double(60), earthRadius* Util::deg2Rad(beagleData_(1)) / double(60) * aspectRatio;

		//std::cout << "XYInit "<< xyInit(0) << " - " << xyInit(1) << std::endl;
		_beagleMeas << 0.0, 0.0, Util::deg2Rad(beagleData_(2)), Util::deg2Rad(beagleData_(3));
		beagleInit = false;
	}
	else {
		//Compute position of Beagle relative to first Beagle position
		//Lat-lon data is close to 0 - 0, so no subtraction of initial position necessary, but is done for completeness
		_beagleMeas << earthRadius* Util::deg2Rad(beagleData_(0)) / double(60.0) - xyInit(0), earthRadius* Util::deg2Rad(beagleData_(1)) / double(60.0) * aspectRatio - xyInit(1), Util::deg2Rad(beagleData_(2)), Util::deg2Rad(beagleData_(3))/double(60.0);
		//std::cout <<"LAT -LON "<< beagleData_(0) << " - " << beagleData_(1) << std::endl;

	}
}

void DataAss::setTargetData(Eigen::Vector4d& targetData_) {

	//Compute position of Beagle relative to first Beagle position
	//Lat-lon data is close to 0 - 0, so no subtraction of initial position necessary, but is done for completeness
	_targetMeas << earthRadius* Util::deg2Rad(targetData_(0)) / double(60.0) - xyInit(0), earthRadius* Util::deg2Rad(targetData_(1)) / double(60.0) * aspectRatio - xyInit(1), Util::deg2Rad(targetData_(2)), Util::deg2Rad(targetData_(3))/double(60.0);
	//std::cout <<"LAT -LON "<< beagleData_(0) << " - " << beagleData_(1) << std::endl;
}

std::vector<int> DataAss::distanceDet(const detection& info, std::vector<prediction> rdet) {
	Eigen::MatrixXd d(rdet.size(), info.cameraAngle.size());
	double stdDev = 18;
	double mu = 80; //Should be dependent on range

	for (int i = 0; i < rdet.size(); i++) {
		for (int j = 0; j < info.cameraAngle.size(); j++) {
			d(i, j) = std::min(abs(info.cameraAngle[j] - rdet[i].angle) * pow(rdet[i].range,1.2)/ (1 / (sqrt(2.0 * M_PI)*stdDev)*std::exp(-0.5*pow((info.cameraElevation[j] - mu) / stdDev, 2))),1e4);
			
			if (abs(info.cameraAngle[j] - rdet[i].angle) > angleMatchThres)
				d(i, j) = 10000;
		}
	}

	std::cout << d << std::endl;
	vector<vector<int>> ans = Util::makeCombi(d.cols()-1, d.rows());
	if (ans.empty()) {
		//In case of only one detection
		vector<int> match(d.rows(),-1);
		std::ptrdiff_t idx;
		float minOfN = d.col(0).minCoeff(&idx);
		if (d(idx,0) != 10000)
			match[idx] = 0;
		return match;
	}
	else {
		for (int i = 0; i < ans.size(); i++) {
			for (int j = 0; j < ans[i].size(); j++) {
				cout << ans.at(i).at(j) << " ";
			}
			cout << endl;
		}

		Eigen::VectorXd n(ans.size());
		int count = 0;
		for (auto&& comb : ans) {
			double sum = 0;
			for (int i = 0; i < comb.size(); i++) {
				sum += d(i, comb[i]);
			}
			n(count) = sum;
			count++;
		}

		std::ptrdiff_t idx;
		float minOfN = n.minCoeff(&idx);

		std::vector<double> angles;
		for (auto&& r : rdet) {
			angles.push_back(r.angle);
		}

		for (int i = 0; i < ans[idx].size(); i++) {
			//Remove ungated entries
			if (d(i, ans[idx][i]) == 10000) {
				ans[idx][i] = -1;
			}
			//Remove optical blocked vessels
			std::pair<bool, int> crossVessel = findRangeVector(angles, angles[i], angleMatchThres + Util::deg2Rad(1), i);
			if (crossVessel.first && rdet[i].range > rdet[crossVessel.second].range)
				ans[idx][i] = -1;
		}

		for (int j = 0; j < ans[idx].size(); j++) {
			cout << ans[idx].at(j) << " ";
		}
		cout << endl;

		return ans[idx];
	}
}

std::pair<vector<double>,vector<double>> DataAss::distanceDet(vector<double> cdet, vector<double> hdet, double rdet) {
	std::pair<vector<double>, vector<double>> result;
	vector<double> d;
	double stdDev = 18;
	double mu = 80; //Should be dependent on range
	vector<double> lambda;
	vector<double> probability;

	for (int i = 0; i < cdet.size(); i++) {
		lambda.push_back(1 / (sqrt(2.0 * M_PI)*stdDev)*std::exp(-0.5*pow((hdet[i] - mu) / stdDev, 2)));
		d.push_back(abs(cdet[i] - rdet));
		probability.push_back(std::min(d.back() / lambda.back(), 1e6));
	}
	result.first = d;
	result.second = probability;
	return result;
}

vector<double> DataAss::distancePL(matchedDetections detection, prediction prdct) {
	vector<double> d;

	for (int i = 0; i < detection.relRange.size(); i++) {
		d.push_back(sqrt(pow(detection.relRange[i], 2) + pow(prdct.range, 2) - 2 * detection.relRange[i] * prdct.range*cos(Util::deg2Rad(prdct.angle - detection.relAngle[i]))));
	}
	return d;
}

vector<int> DataAss::distancePL(matchedDetections detection, std::vector<prediction> prdct) {
	Eigen::MatrixXd d(prdct.size(), detection.relRange.size());
	//vector<double> d;
	for (int i = 0; i < prdct.size(); i++) {
		for (int j = 0; j < detection.relRange.size(); j++) {
			d(i, j) = sqrt(pow(detection.relRange[j], 2) + pow(prdct[i].range, 2) - 2 * detection.relRange[j] * prdct[i].range*cos(prdct[i].angle - detection.relAngle[j]));
			if (d(i, j) > detectionMatchThres) 
				d(i, j) = 1000;
		}
	}
	std::cout << "d_rad: \n" << d << std::endl;
	vector<vector<int>> ans = Util::makeCombi(d.cols()-1, d.rows());
	
	for (int i = 0; i < ans.size(); i++) {
		for (int j = 0; j < ans[i].size(); j++) {
			cout << ans.at(i).at(j) << " ";
		}
		cout << endl;
	}

	if (ans.empty()) {
		//In case of only one detection
		vector<int> match(d.rows(), -1);
		std::ptrdiff_t idx;
		float minOfN = d.col(0).minCoeff(&idx);
		if (d(idx, 0) != 1000)
			match[idx] = 0;

		std::cout << "MatchRad: ";
		for (int j = 0; j < match.size(); j++) {
			cout << match.at(j) << " ";
		}
		cout << endl;
		return match;
	}
	else {

		Eigen::VectorXd n(ans.size());
		int count = 0;
		for (auto&& comb : ans) {
			double sum = 0;
			for (int i = 0; i < comb.size(); i++) {
				sum += d(i, comb[i]);
			}
			n(count) = sum;
			count++;
		}

		std::ptrdiff_t idx;
		float minOfN = n.minCoeff(&idx);

		for (int i = 0; i < ans[idx].size(); i++) {
			if (d(i, ans[idx][i]) == 1000) {
				ans[idx][i] = -1;
			}
		}

		for (int j = 0; j < ans[idx].size(); j++) {
			cout << ans[idx].at(j) << " ";
		}
		cout << endl;

		return ans[idx];
	}
}

void DataAss::drawResults() {
	//Gather information
	vector<vector<vector<double>>> plotVectorsTargets;
	vector<vector<double>> plotVectorsBeagle;
	vector<string> trackName;

	for (int i = 0; i < tracks_.size(); i++) {
		plotVectorsTargets.push_back(tracks_[i].getPlotVectors());
		std::stringstream dd;
		dd << "Track " << i + 1 << " - Measurement";
		trackName.push_back(dd.str());
		std::stringstream ss;
		ss << "Track " << i + 1 << " - Prediction";
		trackName.push_back(ss.str());
	}
	plotVectorsBeagle = BeagleTrack->getPlotVectors();



	//Plot Tracks
	for (int i = 0; i < plotVectorsTargets.size(); i++) {
		graph.addPlot(plotVectorsTargets[i][0], plotVectorsTargets[i][1], trackName[i * 2]);
		graph.addPlot(plotVectorsTargets[i][2], plotVectorsTargets[i][3], trackName[i * 2 + 1]);
	}

	//Plot Beagle Track
	graph.addPlot(plotVectorsBeagle[0], plotVectorsBeagle[1], "Beagle Measurement");
	graph.addPlot(plotVectorsBeagle[2], plotVectorsBeagle[3], "Beagle Prediction");

	if (!plotVectorsTargets.empty())
		graph.plot();



	//string output = graph.plot(x, y, "y = sqrt(x)");
	//string output = graph.animate(x, y);
	//cout << "Test2:\n" << output << '\n';


	//GnuGraph graph;

	//vector<double> x0, y0, x1, y1;
	//for (size_t i = 0; i < 200; ++i)
	//{
	//	x0.push_back(double(i));
	//	y0.push_back(sqrt(x0[i]));

	//	x1.push_back(double(i));
	//	y1.push_back(pow(x1[i], 1.0 / 3.0));
	//}

	//graph.addPlot(x0, y0, "y = sqrt(x)");
	//graph.addPlot(x1, y1, "y = x^(1/3)");
	////string output = graph.plot();
	////cout << "Test4:\n" << output << '\n';
	//waitKey(0);
}

//template < typename T>
std::pair<bool, int > DataAss::findInVector(const std::vector<double>& vecOfElements, const double& element)
{
	std::pair<bool, int > result;

	// Find given element in vector
	auto it = std::find(vecOfElements.begin(), vecOfElements.end(), element);

	if (it != vecOfElements.end())
	{
		result.second = distance(vecOfElements.begin(), it);
		result.first = true;
	}
	else
	{
		result.first = false;
		result.second = -1;
	}

	return result;
}


std::pair<bool, int > DataAss::findRangeVector(const std::vector<double>& vecOfElements, const double& element, const double& range, const double& elementnr )
{
	std::pair<bool, int > result(false, -1);

	// Find given element in vector
	for (int i = 0; i < vecOfElements.size(); i++) {
		if (i != elementnr) {
			if ((vecOfElements[i] > element - range) && (vecOfElements[i] < element + range)) {
				result.first = true;
				result.second = i;
			}
		}
	}

	return result;
}

vector<vector<Eigen::VectorXd>> DataAss::getStateVectors() {
	vector<vector<Eigen::VectorXd>> stateVectors;
	stateVectors.push_back(BeagleState);
	stateVectors.push_back(TargetState);
	return stateVectors;
}

std::vector<std::vector<std::vector<Eigen::VectorXd>>> DataAss::getResultVectors() {
	std::vector<std::vector<std::vector<Eigen::VectorXd>>> stateVectors;
	for (auto&& track : tracks_) {
		stateVectors.push_back(track.getResultVectors());
	}
	return stateVectors;
}

//void DataAss::NearestNeighbor(detection info){
//	for (int i = 0; i < tracks_.size(); i++) {
//		angleMatchThres = Util::deg2Rad(8 - 6 / double(800)*lastDetection[i]);
//
//			//std::cout << Util::rad2Deg(angleMatchThres) << std::endl;
//			//Match detection after radar update
//			if (!detect.relRange.empty()) {
//				vector<int> ans = distancePL(detect, predictionVector);
//				vector<double> polarDist = distancePL(detect, predictionVector[i]);
//				int idxDetection = min_element(polarDist.begin(), polarDist.end()) - polarDist.begin();
//				//Match if sufficiently near - 
//				//TODO - Write Gating algorithm
//				if (polarDist[idxDetection] < detectionMatchThres) {
//					//Match flag for kalman update
//					matchFlag[i] = 0;
//					tracks_[i].detectionAbsence = 0;
//					tracks_[i].setDetection(detect.relRange[idxDetection], detect.relAngle[idxDetection], detect.relVel[idxDetection], _beaglePrediction, matchFlag[i]);
//					//unassignedDetection.erase(std::remove(unassignedDetection.begin(), unassignedDetection.end(), idxDetection), unassignedDetection.end()); //Check if correct one is erased
//					unassignedDetection[idxDetection] = false;
//				}
//			}
//
//
//			//Match camera detection if no radar update or no gated match
//			if (matchFlag[i] == -1 && !info.cameraAngle.empty()) {
//				/*for (int i = 0; i < info.cameraAngle.size(); i++) {
//					if (info.cameraAngle[i] < 0.097 && info.cameraAngle[i] >  0.095)
//						cv::waitKey(0);
//				}*/
//				std::pair<vector<double>, vector<double>> d = distanceDet(info.cameraAngle, info.cameraElevation, predictionVector[i].angle);
//				int idxMatch = min_element(d.second.begin(), d.second.end()) - d.second.begin();
//				//Match if sufficiently near
//				if ((d.first[idxMatch] < angleMatchThres) && (d.second[idxMatch] < angleMatchThres/1.0e-5)) {
//					//Match flag for kalman update	
//					matchFlag[i] = 1;
//					tracks_[i].detectionAbsence++;
//					//Range prediction is returned as detection - Could be improved if done within tracker
//					tracks_[i].setDetection(predictionVector[i].range, info.cameraAngle[idxMatch], -100, _beaglePrediction, matchFlag[i]);	
//				}
//			}
//
//			//Return prediction as measurement if no match is found
//			if (matchFlag[i] == -1) {
//				matchFlag[i] = 2;
//				tracks_[i].detectionAbsence++;//TODO detectionAbsence - link dt 
//				tracks_[i].setDetection(predictionVector[i].range, predictionVector[i].angle, -100, _beaglePrediction, matchFlag[i]);
//
//			}
//
//			//Terminate track if no radar detection has been received for too long
//			if (tracks_[i].detectionAbsence > absenceThreshold)
//				tracks_.erase(tracks_.begin() + i); //Check if correct one is erased
//		}
//	}