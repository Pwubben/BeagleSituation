#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace cv;



void DataAss::run(detection info) {
	
	//Retrieve predictions and detections from last step
	vector<prediction> predictionVector(tracks_.size());
	vector<double> lastDetection(tracks_.size());
	vector<bool> newRangeDetection(info.radarRange.size(),true);
	for (int i = 0; i < info.radarRange.size(); i++) {
		std::cout << info.radarRange[i] << std::endl;
	}
	//Empty matched detection vector
	detect.relRange.clear();
	detect.relAngle.clear();

	//for (int i = 0; i < info.radarRange.size(); i++) {
	//	//Initiate new detection indexes with all detections
	//	newRangeDetection.push_back(i);
	//}

	for (int i = 0; i < tracks_.size(); i++) {//Change later to more robust form - PDA
		predictionVector[i] = tracks_[i].getPrediction();
		lastDetection[i] = tracks_[i].getDetection();
		
	
		//Check for range update
		std::pair<bool,int> result = findInVector(info.radarRange, lastDetection[i]);
		if (result.first) {
			//Erase detection from new detection vector if it was old
			//newRangeDetection.erase(std::remove(newRangeDetection.begin(), newRangeDetection.end(), result.second), newRangeDetection.end());//Check if this is correct one
			newRangeDetection[result.second] = false;
			std::cout << tracks_[i].getDetection() << " - " << newRangeDetection[result.second] << std::endl;
		}
		
	}
	
	//Match radar and camera detections in case of new radar detection
	for (int i = 0; i < newRangeDetection.size(); i++) {
		if (newRangeDetection[i]) {
			//Assign radar range to matched detection slot
			detect.relRange.push_back(info.radarRange[i]);
			//TODO - vector subscript out of range
			vector<double> d = distanceDet(info.cameraAngle, info.radarAngle[i]);
			int idxMatch = min_element(d.begin(), d.end()) - d.begin();

			bool camMatch = false;
			//Match if sufficiently near - 
			//TODO - Write Gating algorithm
			if (!info.cameraAngle.empty()) {
				if (d[idxMatch] < angleMatchThres) {
					detect.relAngle.push_back(info.cameraAngle[idxMatch]);
					camMatch = true;
				}
			}
			//Otherwise match radar angle
			if (!camMatch) 
				detect.relAngle.push_back(info.radarAngle[i]);
		}
	}

	vector<bool> unassignedDetection(detect.relAngle.size(),true);
	//for (int i = 0; i < detect.relAngle.size(); i++) {
	//	unassignedDetection.push_back(i);
	//}

	//Match detection to track
	for (int i = 0; i < tracks_.size(); i++) {
		bool matchFlag = false;
		//Match detection after radar update
		if (!detect.relRange.empty()) {
			vector<double> polarDist = distancePL(detect, predictionVector[i]);
			int idxDetection = min_element(polarDist.begin(), polarDist.end()) - polarDist.begin();
			//Match if sufficiently near - 
			//TODO - Write Gating algorithm
			if (polarDist[idxDetection] < detectionMatchThres) {
				tracks_[i].setDetection(detect.relRange[idxDetection], detect.relAngle[idxDetection], _beagleMeas.head(3)); 
				//unassignedDetection.erase(std::remove(unassignedDetection.begin(), unassignedDetection.end(), idxDetection), unassignedDetection.end()); //Check if correct one is erased
				unassignedDetection[idxDetection] = false;
				//Reset detection count
				tracks_[i].detectionAbsence = 0;
				matchFlag = true;
			}
		}

		//Match camera detection if no radar update or no gated match
		if (!matchFlag && !info.cameraAngle.empty()) {
			vector<double> d = distanceDet(info.cameraAngle, predictionVector[i].angle);
			int idxMatch = min_element(d.begin(), d.end()) - d.begin();
			//Match if sufficiently near
			if (d[idxMatch] < angleMatchThres) {
				tracks_[i].detectionAbsence++;
				//Range prediction is returned as detection - Could be improved if done within tracker
				tracks_[i].setDetection(predictionVector[i].range, info.cameraAngle[idxMatch], _beagleMeas.head(3));
				matchFlag = true;
			}
		}
		//Return prediction as measurement if no match is found
		if (!matchFlag) {
			tracks_[i].detectionAbsence++;//TODO detectionAbsence - link dt 
			tracks_[i].setDetection(predictionVector[i].range, predictionVector[i].angle, _beagleMeas.head(3));
		}

		//Terminate track if no radar detection has been received for too long
		//if (tracks_[i].detectionAbsence > absenceThreshold)
			//tracks_.erase(tracks_.begin() + i); //Check if correct one is erased
	}

	//Initiate track if detection is not assigned
	for (int i = 0; i < unassignedDetection.size(); i++) {
		if (unassignedDetection[i]) {
			tracks_.push_back(Track(detect.relRange[i], detect.relAngle[i], _beagleMeas.head(3)));
		}
	}

	//TODO Beagle KF - Obtain Beagle updates 
	BeagleTrack->compute(_beagleMeas);
	_beaglePrediction = BeagleTrack->getBeaglePrediction();

	//Run each track for new predictions
	for (int i = 0; i < tracks_.size(); i++) {
		tracks_[i].run(_beaglePrediction);
	}
}

void DataAss::setBeagleData(Eigen::Vector4f& beagleData_) {
	//Set initial Beagle position as reference point - if first
	if (beagleInit) {
		//Convert lon-lat to x-y
		aspectRatio = cos(beagleData_(0));
		xyInit << earthRadius* Util::deg2Rad(beagleData_(0) / float(100)), earthRadius* Util::deg2Rad(beagleData_(1) / float(100)) * aspectRatio;
		_beagleMeas << 0.0, 0.0, Util::deg2Rad(beagleData_(2)), Util::deg2Rad(beagleData_(3));
		beagleInit = false;
	}
	else {
		//Compute position of Beagle relative to first Beagle position
		//Lat-lon data is close to 0 - 0, so no subtraction of initial position necessary, but is done for completeness
		_beagleMeas << earthRadius* Util::deg2Rad(beagleData_(0) / float(100)) - xyInit(0), earthRadius* Util::deg2Rad(beagleData_(1) / float(100)) * aspectRatio - xyInit(1), Util::deg2Rad(beagleData_(2)), Util::deg2Rad(beagleData_(3));
	}
}

vector<double> DataAss::distanceDet(vector<double> cdet, double rdet) {
	vector<double> d;
	for (int i = 0; i < cdet.size(); i++) {
		d.push_back(abs(cdet[i] - rdet));
	}
	return d;
}
vector<double> DataAss::distancePL(matchedDetections detection, prediction prdct) {
	vector<double> d;
	for (int i = 0; i < detection.relRange.size(); i++) {
		d.push_back(sqrt(pow(detection.relRange[i],2)+pow(prdct.range,2)-2* detection.relRange[i]* prdct.range*cos(prdct.angle-detection.relAngle[i])));
	}
	return d;
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


