#include "stdafx.h"
#include "GnuGraph.h"
#include "Tracker.h"
#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace cv;

GnuGraph graph;

void DataAss::run(const detection& info) {
	
	//Retrieve predictions and detections from last step
	vector<prediction> predictionVector(tracks_.size());
	vector<float> lastDetection(tracks_.size());
	vector<bool> newRangeDetection(info.radarRange.size(),true);
	
	for (int i = 0; i < info.radarRange.size(); i++) {
		//std::cout << info.radarRange[i] << std::endl;
	}
	//std::cout << "Tracks: " << tracks_.size() << std::endl;
	//Empty matched detection vector
	detect.relRange.clear();
	detect.relAngle.clear();

	for (int i = 0; i < info.radarRange.size(); i++) {
		//std::cout << "Range: " << info.radarRange[i] << "- Angle: " << info.radarAngle[i] << std::endl;
	}

	for (int i = 0; i < tracks_.size(); i++) {//Change later to more robust form - PDA
		predictionVector[i] = tracks_[i].getPrediction();
		lastDetection[i] = tracks_[i].getDetection();
		
	
		//Check for range update
		//std::pair<bool,int> result = findInVector(info.radarRange, lastDetection[i]);
		std::pair<bool, int> result = findRangeVector(info.radarRange, lastDetection[i], 2.0);

		if (result.first ) {
			//Erase detection from new detection vector if it was old
			//newRangeDetection.erase(std::remove(newRangeDetection.begin(), newRangeDetection.end(), result.second), newRangeDetection.end());//Check if this is correct one
			newRangeDetection[result.second] = false;
			//std::cout << tracks_[i].getDetection() << " - " << newRangeDetection[result.second] << std::endl;
		}
	}

	//Match radar and camera detections in case of new radar detection
	for (int i = 0; i < newRangeDetection.size(); i++) {
		if (newRangeDetection[i]) {
			//
			//Assign radar range to matched detection slot
			detect.relRange.push_back(info.radarRange[i]);
			//TODO - vector subscript out of range
			vector<float> d = distanceDet(info.cameraAngle, info.radarAngle[i]);
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

	vector<int> matchFlag(tracks_.size(),-1);
	
	radarCount++;

	//Match detection to track
	for (int i = 0; i < tracks_.size(); i++) {
		//Match detection after radar update
		if (!detect.relRange.empty()) {
			vector<float> polarDist = distancePL(detect, predictionVector[i]);
			int idxDetection = min_element(polarDist.begin(), polarDist.end()) - polarDist.begin();
			//Match if sufficiently near - 
			//TODO - Write Gating algorithm
			if (polarDist[idxDetection] < detectionMatchThres) {
				//Skip if it is first two new detections 
				if (radarCount > 1) {
					//std::cout << "RadarCount: = " << radarCount << std::endl;
					radarCount = -3;
					unassignedDetection[idxDetection] = false;
				}
				else if (radarCount == 1) {
					tracks_[i].detectionAbsence = 0;
					tracks_[i].setDetection(detect.relRange[idxDetection], detect.relAngle[idxDetection], _beaglePrediction);
					//unassignedDetection.erase(std::remove(unassignedDetection.begin(), unassignedDetection.end(), idxDetection), unassignedDetection.end()); //Check if correct one is erased
					unassignedDetection[idxDetection] = false;
					//Match flag for kalman update
					matchFlag[i] = 0;
				}
			}				
		}

		//Match camera detection if no radar update or no gated match
		//if (matchFlag[i] == -1 && !info.cameraAngle.empty()) {
		//	vector<float> d = distanceDet(info.cameraAngle, predictionVector[i].angle);
		//	int idxMatch = min_element(d.begin(), d.end()) - d.begin();
		//	//Match if sufficiently near
		//	if (d[idxMatch] < angleMatchThres) {
		//		tracks_[i].detectionAbsence++;
		//		//Range prediction is returned as detection - Could be improved if done within tracker
		//		tracks_[i].setDetection(predictionVector[i].range, info.cameraAngle[idxMatch], _beaglePrediction);
		//		//Match flag for kalman update
		//		matchFlag[i] = 1;
		//	}
		//}

		//Return prediction as measurement if no match is found
		if (matchFlag[i] == -1) {
			tracks_[i].detectionAbsence++;//TODO detectionAbsence - link dt 
			tracks_[i].setDetection(predictionVector[i].range, predictionVector[i].angle, _beaglePrediction);
			matchFlag[i] = 2;
		}

		//Terminate track if no radar detection has been received for too long
		if (tracks_[i].detectionAbsence > absenceThreshold)
			tracks_.erase(tracks_.begin() + i); //Check if correct one is erased
	}

	//Initiate track if detection is not assigned
	for (int i = 0; i < unassignedDetection.size(); i++) {
		if (unassignedDetection[i] && tracks_.size() < 1) {
			tracks_.push_back(Track(detect.relRange[i], detect.relAngle[i], _beaglePrediction));
			matchFlag.push_back(0);
		}
	}

	//TODO Beagle KF - Obtain Beagle updates 
	BeagleTrack->compute(_beagleMeas);
	_beaglePrediction = BeagleTrack->getBeaglePrediction(); //This prediction is correct for the track runs, for the previously used ones might have to use updated state

	//Run each track for new predictions
	for (int i = 0; i < tracks_.size(); i++) {
		tracks_[i].run(_beaglePrediction, matchFlag[i]);
	}

	drawCount++;
	//Draw results
	if (drawCount > 300) {
		drawResults();
		drawCount = 0;
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

vector<float> DataAss::distanceDet(vector<float> cdet, float rdet) {
	vector<float> d;
	for (int i = 0; i < cdet.size(); i++) {
		d.push_back(abs(cdet[i] - rdet));
	}
	return d;
}
vector<float> DataAss::distancePL(matchedDetections detection, prediction prdct) {
	vector<float> d;
	for (int i = 0; i < detection.relRange.size(); i++) {
		d.push_back(sqrt(pow(detection.relRange[i],2)+pow(prdct.range,2)-2* detection.relRange[i]* prdct.range*cos(Util::deg2Rad(prdct.angle-detection.relAngle[i]))));
	}
	return d;
}

void DataAss::drawResults() {
	//Gather information
	vector<vector<vector<float>>> plotVectorsTargets;
	vector<vector<float>> plotVectorsBeagle;
	vector<string> trackName;
	
	for (int i = 0; i < tracks_.size(); i++) {
		plotVectorsTargets.push_back(tracks_[i].getPlotVectors());
		std::stringstream dd;
		dd << "Track " << i+1 << " - Measurement";
		trackName.push_back(dd.str());
		std::stringstream ss;
		ss << "Track " << i+1 << " - Prediction";
		trackName.push_back(ss.str());
	}
	plotVectorsBeagle = BeagleTrack->getPlotVectors();

	

	//Plot Tracks
	for (int i = 0; i < plotVectorsTargets.size(); i++) {
		graph.addPlot(plotVectorsTargets[i][0], plotVectorsTargets[i][1], trackName[i*2]);
		graph.addPlot(plotVectorsTargets[i][2], plotVectorsTargets[i][3], trackName[i*2+1]);
	}

	//Plot Beagle Track
	graph.addPlot(plotVectorsBeagle[0], plotVectorsBeagle[1], "Beagle Measurement");
	graph.addPlot(plotVectorsBeagle[2], plotVectorsBeagle[3], "Beagle Prediction");

	if(!plotVectorsTargets.empty())
		graph.plot();



	//string output = graph.plot(x, y, "y = sqrt(x)");
	//string output = graph.animate(x, y);
	//cout << "Test2:\n" << output << '\n';


	//GnuGraph graph;

	//vector<float> x0, y0, x1, y1;
	//for (size_t i = 0; i < 200; ++i)
	//{
	//	x0.push_back(float(i));
	//	y0.push_back(sqrt(x0[i]));

	//	x1.push_back(float(i));
	//	y1.push_back(pow(x1[i], 1.0 / 3.0));
	//}

	//graph.addPlot(x0, y0, "y = sqrt(x)");
	//graph.addPlot(x1, y1, "y = x^(1/3)");
	////string output = graph.plot();
	////cout << "Test4:\n" << output << '\n';
	//waitKey(0);
}

//template < typename T>
std::pair<bool, int > DataAss::findInVector(const std::vector<float>& vecOfElements, const float& element)
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


std::pair<bool, int > DataAss::findRangeVector(const std::vector<float>& vecOfElements, const float& element, const float& range)
{
	std::pair<bool, int > result(false,-1);

	// Find given element in vector
	for (int i = 0; i < vecOfElements.size(); i++) {
		if ((vecOfElements[i] > element - range) && (vecOfElements[i] < element + range)) {
			result.first = true;
			result.second = i;
		}
	}

	return result;
}

