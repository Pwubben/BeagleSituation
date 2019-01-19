#include "stdafx.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
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
	vector<bool> newRangeDetection(tracks_.size(),false);

	for (int i = 0; i < tracks_.size(); i++) {//Change later to more robust form - PDA
		predictionVector[i] = tracks_[i].getPrediction();
		lastDetection[i] = tracks_[i].getDetection();
		//Check for range update
		if (std::find(info.radarRange.begin(), info.radarRange.end(), lastDetection[i]) == info.radarRange.end()) {
			newRangeDetection[i] = true;
		}
	}
	
	//Match radar and camera detections
	for (int i = 0; i < info.radarRange.size(); i++) {
		
		//Assign radar range to matched detection slot
		detect.relRange.push_back(info.radarRange[i]);
		vector<double> d = distance(info.cameraAngle, info.radarAngle[i]);
		int idxMatch = min_element(d.begin(), d.end()) - d.begin();

		//Match if sufficiently near
		if (abs(info.cameraAngle[idxMatch] - info.radarAngle[i]) < angleMatchThres) {
			detect.relAngle.push_back(info.cameraAngle[idxMatch]);
		}
		//Otherwise match radar angle
		else
			detect.relAngle.push_back(info.radarAngle[i]);
	}
}

vector<double> distance(vector<double> cdet, double rdet) {
	vector<double> d;
	for (int i = 0; i < cdet.size(); i++) {
		d.push_back(abs(cdet[i] - rdet));
	}
	return d;
}
vector<double> distancePL(matchedDetections detection, double prediction) {
	vector<double> d;
	for (int i = 0; i < cdet.size(); i++) {
		d.push_back(abs(cdet[i] - rdet));
	}
	return d;
}
