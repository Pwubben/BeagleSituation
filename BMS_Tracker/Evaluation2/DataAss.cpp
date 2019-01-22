#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace cv;

struct matchedDetections {
	vector<double> relRange;
	vector<double> relAngle;
}detect;

void DataAss::run(detection info) {
	
	//Retrieve predictions and detections from last step
	vector<prediction> predictionVector(tracks_.size());
	vector<double> lastDetection(tracks_.size());
	vector<int> newRangeDetection;

	for (int i = 0; i < tracks_.size(); i++) {//Change later to more robust form - PDA
		predictionVector[i] = tracks_[i].getPrediction();
		lastDetection[i] = tracks_[i].getDetection();
		//Check for range update
		std::pair<bool,int> result = findInVector(info.radarRange, lastDetection[i]);
		if (result.first) {
			newRangeDetection.push_back(result.second);
		}
	}
	
	//Match radar and camera detections in case of new radar detection
	for (int i = 0; i < newRangeDetection.size(); i++) {
		//Assign radar range to matched detection slot
		detect.relRange.push_back(info.radarRange[newRangeDetection[i]]);
		vector<double> d = distance(info.cameraAngle, info.radarAngle[newRangeDetection[i]]);
		int idxMatch = min_element(d.begin(), d.end()) - d.begin();

		//Match if sufficiently near - Write Gating algorithm
		if (d[idxMatch] < angleMatchThres) {
			detect.relAngle.push_back(info.cameraAngle[idxMatch]);
		}
		//Otherwise match radar angle
		else
			detect.relAngle.push_back(info.radarAngle[newRangeDetection[i]]);
	}

	vector<int> unassignedDetection;
	for (int i = 0; i < detect.relAngle.size(); i++) {
		unassignedDetection.push_back(i);
	}

	//Match detection to track
	for (int i = 0; i < tracks_.size(); i++) {
		bool matchFlag = false;
		//Match detection after radar update
		if (!detect.relRange.empty()) {
			vector<double> polarDist = distancePL(detect, predictionVector[i]);
			int idxDetection = min_element(polarDist.begin(), polarDist.end()) - polarDist.begin();
			//Match if sufficiently near - Write Gating algorithm
			if (polarDist[idxDetection] < detectionMatchThres) {
				tracks_[i].setDetection(detect.relRange[idxDetection], detect.relAngle[idxDetection]);
				unassignedDetection.erase(unassignedDetection.begin() + idxDetection); //Check if correct one is erased

				//Reset detection count
				tracks_[i].detectionAbsence = 0;
				matchFlag = true;
			}
		}
		//Match camera detection if no radar update or no gated match
		if (!matchFlag && !info.cameraAngle.empty()) {
			vector<double> d = distance(info.cameraAngle, predictionVector[i].angle);
			int idxMatch = min_element(d.begin(), d.end()) - d.begin();
			//Match if sufficiently near
			if (d[idxMatch] < angleMatchThres) {
				tracks_[i].setDetection(0, info.cameraAngle[idxMatch]);
				matchFlag = true;
			}
		}
		//Assign zero's if no match is found
		if (!matchFlag) {
			tracks_[i].setDetection(0, 0);
			tracks_[i].detectionAbsence++;
		}
		//Terminate track if no detection has been received for too long
		if (tracks_[i].detectionAbsence > 30)
			tracks_.erase(tracks_.begin() + i); //Check if correct one is erased
	}

	//Initiate track if detection is not assigned
	for (int i = 0; i < unassignedDetection.size(); i++) {
		tracks_.push_back(Track(detect.relRange[unassignedDetection[i]], detect.relAngle[unassignedDetection[i]]));
	}

	for (int i = 0; i < tracks_.size(); i++) {
		tracks_[i].run();
	}
}

vector<double> distance(vector<double> cdet, double rdet) {
	vector<double> d;
	for (int i = 0; i < cdet.size(); i++) {
		d.push_back(abs(cdet[i] - rdet));
	}
	return d;
}
vector<double> distancePL(matchedDetections detection, prediction prdct) {
	vector<double> d;
	for (int i = 0; i < detection.relRange.size(); i++) {
		d.push_back(sqrt(pow(detection.relRange[i],2)+pow(prdct.range,2)-2* detection.relRange[i]* prdct.range*cos(prdct.angle-detection.relAngle[i])));
	}
	return d;
}

//template < typename T>
std::pair<bool, int > findInVector(const std::vector<double>  & vecOfElements, const double  & element)
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