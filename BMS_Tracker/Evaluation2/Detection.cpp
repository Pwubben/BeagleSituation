
#include "Tracker.h"
#include "BMS.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;



void Detection::run(std::string File, std::string groundTruthFile, std::string beagleFile, std::string radarFile, std::string targetFile, std::string beagleDes, std::string targetDes) {
	cout << File << endl;
	//Load ground truth data
	Rect GT;
	/*std::vector<std::vector<int>> GroundTruth = readGroundTruth(getFileString(groundTruthFile));

	for (int s = 0; s < GroundTruth.size(); s++) {
		Rect coord(GroundTruth[s][0], GroundTruth[s][1], GroundTruth[s][2], GroundTruth[s][3]);
		GT.push_back(coord);
	}*/
	
	//Load Beagle Data
	std::vector<Eigen::Vector4d> beagleData_ = loadBeagleData(getFileString(beagleFile));
	std::vector<Eigen::Vector4d> targetData_ = loadTargetData(getFileString(targetFile));
	std::vector<Eigen::Vector3d> radarData_  = loadRadarData(getFileString(radarFile));
	int count = 0;

	//Performance parameters
	double max_dimension = 800;
	double sample_step = 25;
	double threshold = 5.5;

	bool check(false);
	try {
		if (!check) {

			cv::VideoCapture capture(getFileString(File));
			Mat src;

			capture >> src;
			//Function storing window information - aanpassen
			windowDetect(src,max_dimension); //radarScreenDetect()

			while (1) {
				double duration = static_cast<double>(cv::getTickCount());
				//if (count == 0) {
				//	for (int i = 0; i < 110; i++) {
				//		capture >> src;
				//		count++;
				//	}
				//}

				capture >> src;
				if (src.empty())
					break;

				info.radarRange.clear();
				info.radarAngle.clear();
				info.radarVel.clear();
				info.cameraAngle.clear();
				info.cameraElevation.clear();

				//Radar detector
				//radarDetection(src(radarWindow));
				info.radarRange.push_back(radarData_[count](0));
				info.radarAngle.push_back(radarData_[count](1));
				info.radarVel.push_back(radarData_[count](2));

				//Camera detector
				saliencyDetection(src, max_dimension, sample_step, threshold, GT);
		
				data_ass_->setBeagleData(beagleData_[count]);
				data_ass_->setTargetData(targetData_[count]);

				data_ass_->run(info);
				count++;
				duration = static_cast<double>(cv::getTickCount()) - duration;
				duration /= cv::getTickFrequency();
				//std::cout << duration << std::endl;

				/*if (count == stopFrame)
					break;*/
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
	//std::cout << "ret (python)  = " << std::endl << format(data, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

	vector<vector<Eigen::VectorXd>> stateVectors = data_ass_->getStateVectors();

	//writeDataFile(stateVectors, getFileString(beagleDes), getFileString(targetDes));
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
	seaWindow = cv::Rect(10, radarCenter.y + radarRadius + 70, src.cols - 10, src.rows - radarCenter.y - radarRadius - 70);
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

	if (location.size() > 1) {
		std::cout << 2 << std::endl;
	}
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

void Detection::saliencyDetection(Mat src, double max_dimension, double sample_step, double threshold, Rect GT)
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

	for (int i = 0; i < boundRect.size(); i++) {
		boundRect[i].x = boundRect[i].x*(double)(src.cols / (double)(max_dimension*src.cols/maxD));
		boundRect[i].width = boundRect[i].width*(double)(src.cols / (double)(max_dimension*src.cols / maxD));
		boundRect[i].y = boundRect[i].y*(double)(src.rows / (double)(max_dimension*src.rows / src.cols));
		boundRect[i].height = boundRect[i].height*(double)(src.rows / (double)(max_dimension*src.rows / src.cols));
	
		//detectionAngles.push_back(double((boundRect[i].x + 0.5*boundRect[i].width) * double(FOV / src.cols)));
		angle = double((boundRect[i].x + 0.5*boundRect[i].width - 0.5*src.cols) * double(FOV / src.cols));
		/*if (angle < 0)
			angle += 2.0* M_PI;*/
		info.cameraAngle.push_back(angle);
		info.cameraElevation.push_back(boundRect[i].y + boundRect[i].height/2);

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

	//color = Scalar(0, 0, 200);
	/*cv::Point2i point(GT.x + 40, GT.y -383);
	cout << point.x << " - " << point.y + GT.height/2 << endl;
	rectangle(drawWindow, GT.tl()+cv::Point2i(40, - 383), GT.br() + cv::Point2i(40, -383), color);*/

	imshow("Src", drawWindow);
	waitKey(1);
	waitKey(1);
	//Compute angle
	
	//info.cameraAngle = detectionAngles;

	//return detectionAngles;
	//Ground truth
	
}

vector<vector<int>> Detection::readGroundTruth(std::string fileName){

	vector<vector<int>> groundTruth;
	ifstream file(fileName);
	string line;
	while (getline(file, line))
	{
		vector<int> row;
		stringstream iss(line);
		string val;

		// while getline gives correct result
		while (getline(iss, val, ','))
		{
			row.push_back(stoi(val));
		}
		groundTruth.push_back(row);
	}

	return groundTruth;
}

std::string Detection::getFileString(std::string fileName) {
	std::string path = "F:\\Nautis Run 5-3-19\\";
	std::stringstream ss;
	ss << path << fileName;
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

std::vector<Eigen::Vector3d> Detection::loadRadarData(std::string radarFile) {
	vector<Eigen::Vector3d> radarData_;
	ifstream file(radarFile);
	string line;
	while (getline(file, line))
	{
		//vector<int> row;
		Eigen::Vector3d row;
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