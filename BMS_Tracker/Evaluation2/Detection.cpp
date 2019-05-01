
#include "Tracker.h"
#include "BMS.h"
#include "opencv2/opencv.hpp"
#include <random>

using namespace cv;
using namespace std;



void Detection::run(std::string path, std::string File, std::string beagleFile, std::string radarFile, std::string targetFile, std::string beagleDes, std::string targetDes, std::string resultDes, int targets,double fov) {
	path_ = path;

	cout << File << endl;
	//Load ground truth data
	
	std::vector<Eigen::VectorXd> GroundTruth = readGroundTruth("G:\\Afstuderen\\ss1_sc_mbGroundTruth.csv");
	/*
	for (int s = 0; s < GroundTruth.size(); s++) {
		Rect coord(GroundTruth[s][0], GroundTruth[s][1], GroundTruth[s][2], GroundTruth[s][3]);
		GT.push_back(coord);
	}*/
	

	//Load Beagle Data
	std::vector<Eigen::Vector4d> beagleData_ = loadBeagleData(getFileString(beagleFile));
	std::vector<Eigen::Vector4d> targetData_ = loadTargetData(getFileString(targetFile));
	std::vector<Eigen::VectorXd> radarData_  = loadRadarData(getFileString(radarFile),targets);
	int count = 0;

	//Create Radar Dropout
	double dropout = 0; // Chance of radar dropout
	constexpr size_t size = 3000;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution dist(1 - dropout); // bernoulli_distribution takes chance of true n constructor

	std::vector<int> dropoutmask(size);
	std::generate(dropoutmask.begin(), dropoutmask.end(), [&] { return dist(gen); });
	size_t ones = std::count(dropoutmask.begin(), dropoutmask.end(), 1);
	dropoutmask[0] = 1;
	std::cout << ones << std::endl;

	//Performance parameters
	double max_dimension = evalSettings.maxDimension;
	double sample_step = 25;
	FOV = Util::deg2Rad(fov);// ::deg2Rad(88.4);
	double totalDuration = 0;
	bool check(false);
	try {
		if (!check) {

			cv::VideoCapture capture(getFileString(File));
			Mat src;

			capture >> src;
			//Function storing window information - aanpassen
			windowDetect(src,max_dimension); //radarScreenDetect()

			while (1) {
				
				/*if (count == 0) {
					for (int i = 0; i < 2400; i++) {
						capture >> src;
						count++;
					}
				}*/
				/*if (count == 1800)
					cv::waitKey(0);*/

				capture >> src;
				if (src.empty())
					break;
				double duration = static_cast<double>(cv::getTickCount());
				info.radarRange.clear();
				info.radarAngle.clear();
				info.radarVel.clear();
				info.cameraAngle.clear();
				info.cameraElevation.clear();

				//Radar detector
				//radarDetection(src(radarWindow));
				for (int i = 0; i < radarData_[0].size(); i+=3) {
					info.radarRange.push_back(radarData_[count](i));
					info.radarAngle.push_back(radarData_[count](i+1));
					info.radarVel.push_back(radarData_[count](i+2));
				}
				info.radarDropout = dropoutmask[count];
				//Camera detector
				if(evalSettings.cameraUtil)
					saliencyDetection(src, max_dimension, sample_step, threshold, GroundTruth[0]);
		
				data_ass_->setBeagleData(beagleData_[count]);
				data_ass_->setTargetData(targetData_[count]);

				data_ass_->run(info);
				count++;
				//std::cout << count << std::endl;
				duration = static_cast<double>(cv::getTickCount()) - duration;
				duration /= cv::getTickFrequency();
				totalDuration += duration;
				//std::cout << duration << std::endl;

				//if (count == 1208)
				//	break;
			}
		}
		else {
			check = false;
		}
	}
	catch (std::exception e) {
		check = true;
		std::cout << e.what() << std::endl;
	}

	/*std::cout << "OverallDuration: " << totalDuration / double(count);
	std::cout << "IMMDuration: " << data_ass_->getDuration() / double(count);*/

	//std::cout << "ret (python)  = " << std::endl << format(data, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

	/*vector<vector<Eigen::VectorXd>> stateVectors = data_ass_->getStateVectors();
	writeDataFile(stateVectors, getFileString(beagleDes), getFileString(targetDes));*/

	vector<vector<vector<Eigen::VectorXd>>> resultVectors = data_ass_->getResultVectors();
	writeResultFile(resultVectors, getFileString(resultDes));

	/*std::ofstream targetFile_("G:\\Afstuderen\\AngleFile_ss1.csv", std::ofstream::out | std::ofstream::trunc);

	for (int i = 0; i < angleVector.size(); i++) {
		targetFile_ << angleVector[i](0) << "," << angleVector[i](1) << "," << angleVector[i](2) << std::endl;
	}
	targetFile_.close();
*/
}

void Detection::windowDetect(cv::Mat src, double max_dimension) {
	cv::Mat src_gray;
	cv::cvtColor(src, src_gray, CV_BGR2GRAY);
	GaussianBlur(src_gray, src_gray, cv::Size(9, 9), 2, 2);

	std::vector<cv::Vec3f> circles;
	cv::HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 1000, 100, 50);

	// Determine circle properties
	radarCenter = { cvRound(circles[0][0]), cvRound(circles[0][1]) };
	radarRadius = cvRound(circles[0][2]);
	// circle center
	//circle(src, radarCenter, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
	// circle outline
	//circle(src, radarCenter, radarRadius, cv::Scalar(0, 0, 255), 3, 8, 0);

	radarWindow = cv::Rect(radarCenter.x - radarRadius, max(0,radarCenter.y - radarRadius), 2 * radarRadius, 2 * radarRadius);
	//seaWindow = cv::Rect(0,src.rows - 532, src.cols-0, 532);

	seaWindow = cv::Rect(0,src.rows - 532, src.cols-0, 532);
	//cv::imshow("Hough", src);
	//cv::waitKey(0);
	radarCenter.x -= radarWindow.x;
	radarCenter.y -= radarWindow.y;

	src = src(seaWindow);

	double w = (double)src.cols, h = (double)src.rows;
	maxD = max(w, h);
	resizeDim = { (int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD) };

}

void Detection::radarDetection(Mat src) {
	// Radar Detection	
	std::vector<cv::Mat> channels_rad;
	cv::split(src, channels_rad);

	//TODO - Temporary value
	radarCenter = cv::Point(154, 151);

	cv::Mat radar_mask;
	cv::threshold(channels_rad[2], radar_mask, 130, 255, CV_THRESH_BINARY);
	imshow("radar", radar_mask);
	waitKey(1);
	circle(src, radarCenter, 1, cv::Scalar(0, 255, 0), -1, 8, 0);
	
	//std::cout << "ret (python)  = " << std::endl << format(radar_img, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy_rad;
	findContours(radar_mask, contours, hierarchy_rad, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

	std::vector<float> radius_detection(contours.size());
	std::vector<Point2f> location(contours.size());
	double range, angle;
	int eraseIdx = -1;
	for (int i = 0; i < contours.size(); i++)
	{
		minEnclosingCircle((Mat)contours[i], location[i], radius_detection[i]);

		for (int i = 0; i < location.size(); i++) {
			if ((abs(double(location[i].x) - radarCenter.x) < 4) && (abs(double(location[i].y) - radarCenter.y) < 4)) {
				if (!centerInit) {
					radarCenter = cv::Point(154, 151);
					//radarCenter = cv::Point(location[i].x, location[i].y);
					centerInit = true;
				}
				eraseIdx = i;
			}
		}
	}

	if (eraseIdx > -1)
		location.erase(location.begin() + eraseIdx);

	
	for (int i = 0; i < location.size(); i++)
	{
		range = sqrt(pow(double(location[i].x- radarCenter.x), 2) + pow(double(radarCenter.y-location[i].y), 2)) / radarRadius * radarRange;
		angle = atan2(double(location[i].x - radarCenter.x) , double(radarCenter.y - location[i].y));
		info.radarRange.push_back(Util::round(range));
		info.radarAngle.push_back(angle);
		//location[i].x = range;
		//location[i].y = angle;
	}

	//return location;
}

void Detection::saliencyDetection(Mat src, double max_dimension, double sample_step, double threshold, Eigen::VectorXd GT)
{
	//cv::VideoWriter video("TurnSaliency_CovTrh.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, src.size(), true)

	//Crop image to separate radar and visuals
	Mat drawGT = src.clone();
	src = src(seaWindow);
	//imshow("sea", src);
	//waitKey(0);
	//Resize image
	resize(src, src_small, resizeDim, 0.0, 0.0, INTER_AREA);

	//Start timing
	double duration = static_cast<double>(cv::getTickCount());

	// Computing saliency 
	BMS bms(src_small, dilation_width_1, use_normalize, handle_border, colorSpace, whitening);
	bms.computeSaliency((double)sample_step);

	sResult = bms.getSaliencyMap();

	// Post-processing 
	if (dilation_width_2 > 0)
		dilate(sResult, sResult, Mat(), Point(-1, -1), dilation_width_2);
	if (blur_std > 0)
	{
		int blur_width = (int)MIN(floor(blur_std) * 4 + 1, 51);
		GaussianBlur(sResult, sResult, Size(blur_width, blur_width), blur_std, blur_std);
	}

	//Mean and standard deviation of result map
	meanStdDev(sResult, mean, std);

	//std::cout << "ret (python)  = " << std::endl << format(result, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

	thr = mean.at<double>(0) + threshold * std.at<double>(0);

	//Thresholding result map
	cv::threshold(sResult, masked_img, thr, 1, THRESH_BINARY);
	masked_img.convertTo(mask_trh, CV_8UC1);

	// Find contours in mask
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	cv::findContours(mask_trh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//vector<vector<Point> > contours_poly(contours.size()); //Remove
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		//approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours[i]));
	}

	//Resize bounding rectangles to compare
	//vector<double> detectionAngles;
	double angle;
	vector<int> pixbdr;
	for (int i = 0; i < boundRect.size(); i++) {
		boundRect[i].x = boundRect[i].x*(double)(src.cols / (double)(max_dimension*src.cols/maxD));
		boundRect[i].width = boundRect[i].width*(double)(src.cols / (double)(max_dimension*src.cols / maxD));
		boundRect[i].y = boundRect[i].y*(double)(src.rows / (double)(max_dimension*src.rows / src.cols));
		boundRect[i].height = boundRect[i].height*(double)(src.rows / (double)(max_dimension*src.rows / src.cols));
	
		//detectionAngles.push_back(double((boundRect[i].x + 0.5*boundRect[i].width) * double(FOV / src.cols)));
		//angle = atan(double((boundRect[i].x + 0.5*boundRect[i].width - 0.5*src.cols) * double(2.0*tan(FOV/double(2.0)) / src.cols)));
		
		angle = atan(double((double(boundRect[i].x) + 0.5*double(boundRect[i].width) - 0.5*double(src.cols)) * double(2.0* tan(FOV / double(2.0)) / double(src.cols))));

		/*if (angle < 0)
			angle += 2.0* M_PI;*/
		info.cameraAngle.push_back(angle);
		info.cameraElevation.push_back(boundRect[i].y + boundRect[i].height/2);
		info.boundRectx.push_back(double(boundRect[i].x) + 0.5*double(boundRect[i].width) - 0.5*double(src.cols));

		pixbdr.push_back(double(boundRect[i].x) + 0.5*double(boundRect[i].width));
	}
	duration = static_cast<double>(cv::getTickCount()) - duration;
	duration /= cv::getTickFrequency();

	//cout << duration << endl;
	// Draw bonding rects 
	Mat drawing = Mat::zeros(mask_trh.size(), CV_8UC3);
	RNG rng(0xFFFFFFFF);
	Scalar color = Scalar(0, 200, 50);

	Mat drawWindow = src.clone();
	for (int i = 0; i < contours.size(); i++)
	{
		rectangle(drawWindow, boundRect[i].tl(), boundRect[i].br(), color);
	}
	color = Scalar(0, 0, 200);
	Rect GTrect(GT(0)-2, 532-GT(1)+20 , GT(2), GT(3));
	//rectangle(drawWindow, GTrect.tl(), GTrect.br(), color);

	double GTangle = atan(double((double(GTrect.x) + 0.5*double(GTrect.width) - 0.5*double(src.cols)) * double(2.0* tan(FOV / double(2.0)) / double(src.cols))));

	
	/*int pixel1 = tan(GT(1)) / (2.0 * tan(FOV / 2) / src.cols) + 0.5 * src.cols;
	int pixel2 = tan(GT(3)) / (2.0 * tan(FOV / 2) / src.cols) + 0.5 * src.cols;
	cv::Point GTPoint1 = { pixel1,80 };
	cv::Point GTPoint2 = { pixel2,80 };
	circle(drawWindow, GTPoint1, 1, cv::Scalar(0, 0, 255), 3, 8, 0);
	circle(drawWindow, GTPoint2, 1, cv::Scalar(0, 0, 255), 3, 8, 0);*/
	//color = Scalar(0, 0, 200);
	/*cv::Point2i point(GT.x + 40, GT.y -383);
	cout << point.x << " - " << point.y + GT.height/2 << endl;
	rectangle(drawWindow, GT.tl()+cv::Point2i(40, - 383), GT.br() + cv::Point2i(40, -383), color);*/

	/*imshow("Src", drawWindow);
	waitKey(1);*/
	if (!empty(info.cameraAngle)) {
		//sort(info.cameraAngle.begin(), info.cameraAngle.end());
		//int idx = closest(info.cameraAngle, GTangle);
		Eigen::VectorXd cameraAngles(info.cameraAngle.size());
		for (int i = 0; i < info.cameraAngle.size(); i++) {
			cameraAngles(i) = abs(info.cameraAngle[i] - GTangle);
		}
		std::ptrdiff_t idx;
		float minOfN = cameraAngles.minCoeff(&idx);
		//auto it = lower_bound(info.cameraAngle.begin(), info.cameraAngle.end(), GTangle, [](double a, double b) { return a > b; });
		/*auto it = FindClosest(info.cameraAngle, GTangle);
		int idx = it - info.cameraAngle.begin();*/
		//int idx = std::distance(info.cameraAngle.begin(), it);
		//int l_idx = std::upper_bound(info.cameraAngle.begin(), info.cameraAngle.end(), GTangle) - info.cameraAngle.begin() - 1;
		if ((abs(info.cameraAngle[idx] - GTangle) < Util::deg2Rad(3)) && (info.cameraElevation[idx] < 120.0)) {
			Eigen::Vector3d angles;
			angles << info.cameraAngle[idx] - GTangle, GTangle, info.cameraAngle[idx];
			angleVector.push_back(angles);
		}
	}
	
		
	//waitKey(1);
	//Compute angle
	
	//info.cameraAngle = detectionAngles;

	//return detectionAngles;
	//Ground truth
	
}

int Detection::closest(std::vector<double> const& vec, double value) {
	auto const it = std::lower_bound(vec.begin(), vec.end(), value);
	if (it == vec.begin()) { return -1; }
	else return *(it - 1);
}

auto FindClosest(const std::vector<double>& v, double value)
{
	// assert(std::is_sorted(v.begin(), v.end(), std::greater<>{}));
	auto it = std::lower_bound(v.begin(), v.end(), value, std::greater<>{});

	if (it == v.begin()) {
		return it;
	}
	else if (it == v.end()) {
		return it - 1;
	}
	else {
		return std::abs(value - *it) < std::abs(value - *(it - 1)) ?
			it : it - 1;
	}
}

vector<Eigen::VectorXd> Detection::readGroundTruth(std::string fileName){

	vector<Eigen::VectorXd> groundTruth;
	ifstream file(fileName);
	string line;
	while (getline(file, line))
	{
		Eigen::VectorXd row(4);
		int i = 0;
		stringstream iss(line);
		string val;

		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			row(i) = stof(val);
			i++;
		}

		groundTruth.push_back(row);
	}

	return groundTruth;
}

std::string Detection::getFileString(std::string fileName) {
	std::stringstream ss;
	ss << path_ << fileName;
	std::string file = ss.str();
	return file;
}

std::vector<Eigen::Vector4d> Detection::loadBeagleData(std::string beagleFile) {
	vector<Eigen::Vector4d> beagleData_;
	ifstream file(beagleFile);
	string line;
	while (getline(file, line))
	{
		//vector<int> row;
		Eigen::Vector4d row;
		int i = 0;
		stringstream iss(line);
		string val;

		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			row(i) = stof(val);
			i++;
		}
		
		beagleData_.push_back(row);
	}
	return beagleData_;
}

std::vector<Eigen::Vector4d> Detection::loadTargetData(std::string targetFile) {
	vector<Eigen::Vector4d> targetData_;
	ifstream file(targetFile);
	string line;
	while (getline(file, line))
	{
		//vector<int> row;
		Eigen::Vector4d row;
		int i = 0;
		stringstream iss(line);
		string val;

		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			row(i) = stof(val);
			i++;
		}

		targetData_.push_back(row);
	}
	return targetData_;
}

std::vector<Eigen::VectorXd> Detection::loadRadarData(std::string radarFile, int targets) {
	vector<Eigen::VectorXd> radarData_;
	ifstream file(radarFile);
	string line;
	while (getline(file, line))
	{
		//vector<int> row;
		Eigen::VectorXd row(3*targets);
		int i = 0;
		stringstream iss(line);
		string val;

		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			row(i) = stof(val);
			i++;
		}

		radarData_.push_back(row);
	}
	return radarData_;
}

void Detection::writeDataFile(std::vector<std::vector<Eigen::VectorXd>> stateVectors, std::string BeagleFile, std::string TargetFile) {
	std::ofstream beagleFile_(BeagleFile, std::ofstream::out | std::ofstream::trunc);
	std::ofstream targetFile_(TargetFile, std::ofstream::out | std::ofstream::trunc);

	for (int i = 0; i < stateVectors[0].size(); i++) {
		beagleFile_ << stateVectors[0][i](0) << "," << stateVectors[0][i](1) << "," << stateVectors[0][i](2) << "," << stateVectors[0][i](3) << "," << stateVectors[0][i](4) << std::endl;
	}
	beagleFile_.close();

	for (int i = 0; i < stateVectors[1].size(); i++) {
		targetFile_ << stateVectors[1][i](0) << "," << stateVectors[1][i](1) << "," << stateVectors[1][i](2) << "," << stateVectors[1][i](3) << "," << stateVectors[1][i](4) << std::endl;
	}
	beagleFile_.close();
	targetFile_.close();

}

void Detection::writeResultFile(std::vector<std::vector<std::vector<Eigen::VectorXd>>> resultVectors, std::string resultFile) {
	std::ofstream resultFile_(resultFile, std::ofstream::out | std::ofstream::trunc);
	
	for (int i = 0; i < resultVectors[0][2].size(); i++) {
		for (int h = 0; h < resultVectors.size(); h++) {
			for (int d = 0; d < resultVectors[h][2][0].size(); d++) {
				resultFile_ << resultVectors[h][2][i](d) << ",";
			}
			if (!resultVectors[h][3].empty()) {
				for (int s = 0; s < resultVectors[h][3][0].size(); s++) {
					resultFile_ << resultVectors[h][3][i](s) << ",";
				}
			}
			if (!resultVectors[h][1].empty()) {
				if (i < resultVectors[h][1].size()) {
					for (int g = 0; g < resultVectors[h][1][0].size(); g++) {
						resultFile_ << resultVectors[h][1][i](g) << ",";
					}
				}
				else if (i >= resultVectors[h][1].size()) {
					for (int g = 0; g < resultVectors[h][1][0].size(); g++) {
						resultFile_ << ",";
					}
				}
			}
			if (i < resultVectors[h][0].size()) {
				for (int j = 0; j < resultVectors[h][0][0].size(); j++) {
					resultFile_ << resultVectors[h][0][i](j) << ",";
				}
			}
			else if (i >= resultVectors[h][0].size()) {
				for (int g = 0; g < resultVectors[h][0][0].size(); g++) {
					resultFile_ << ",";
				}
			}
		}
		resultFile_ << std::endl;
	}

	resultFile_.close();

}