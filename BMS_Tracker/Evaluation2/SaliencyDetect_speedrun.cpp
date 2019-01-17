#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include "lbp.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/opencv_modules.hpp"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"

#include <opencv2/core/opengl.hpp>
//#include "opencv2/cudabgsegm.hpp"

#include "trim.h"
#include "BMS.h"
#include "RadarScreenDetect.h"

using namespace cv;
using namespace std;

void SaliencyDetect(cv::VideoCapture capture, vector<vector<Rect>> &boundRectVec,double &avg_time, double max_dimension, double sample_step,double stdThres, vector<Rect> GT,int GT_offset)
{

	int dilation_width_1 = 3;
	int dilation_width_2 = 3;
	float blur_std = 3;
	bool use_normalize = 1;
	bool handle_border = 0;
	int colorSpace = 1;
	bool whitening = 0;

	bool saveVid = false;
	bool useHorizon = false;

	Mat src;
	capture.read(src);

	// Radar and sea windows
	cv::Rect sea_scr;
	cv::Rect radar_scr;

	RadarScreenDetect(src, radar_scr, sea_scr);

	cv::Mat radar_src = src(radar_scr);
	src = src(sea_scr);

	// Preprocessing
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w, h);

	//max_dimension = src.cols;

	//Create output video file
	cv::VideoWriter video("TurnSaliency_CovTrh.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, src.size(), true);

	// Timing variables
	double total_time = 0.0, count = 0.0, duration;
	int  loopcount = 0, GTcount = 0;

	Mat left_vec, right_vec;
	Point maxL, maxR, minL, minR;
	double maxValL, minValL, maxValR, minValR;

	// Thresholding variables
	double trh = 180;
	Mat mean, std;
	Mat src_gray, src_small;
	Size resize_size = { (int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD) };

	while (1)
	{
		capture >> src;

		if (src.empty())
		{
			// Reach end of the video file
			break;
		}

		src = src(sea_scr);
		// Resize image
		Mat src_cr = src;
		if (useHorizon) {

			src_cr = src(sea_scr);
		}
		h = (float)src_cr.rows;
		maxD = max(w, h);
		
		resize(src_cr, src_small, Size((int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD)), 0.0, 0.0, INTER_AREA);

		duration = static_cast<double>(cv::getTickCount());

		// Horizon detection
	
	/*	cvtColor(src_cr, src_gray, CV_BGR2GRAY);
		GaussianBlur(src_gray, src_gray, { 5,5 }, 6);
		imshow("srcgray", src_gray);
		Sobel(src_gray(Range::all(), Range(0, 8)), left_vec, CV_32F, 0, 1, 9);
		Sobel(src_gray(Range::all(), Range(src_gray.cols - 12, src_gray.cols - 4)), right_vec, CV_32F, 0, 1, 9);

		cv::minMaxLoc(left_vec, &minValL, &maxValL, &minL, &maxL);
		cv::minMaxLoc(right_vec, &minValR, &maxValR, &minR, &maxR);
		Point HorL = minL;
		Point HorR = minR;
		if (-minValL < maxValL) {
			HorL = maxL;
		}
		if (-minValR < maxValR) {
			HorR = maxR;
		}
		HorR.x = src_gray.cols - 3;
*/
		/*std::vector<Mat> channels;
		split(src_cr, channels);
		Mat dst;
		Mat flag_mask = (channels[2] < 200);
		imshow("flag", flag_mask);
		channels[2].copyTo(dst, flag_mask);
		channels.erase(channels.begin() + 2);
		channels.push_back(dst);
		merge(channels, src_cr);
		imshow("red", dst);*/



		// LBP computation
		bool incorporate_lbp = false;

		if (incorporate_lbp) {
			Mat lbp_feature, lbp_src;
			cvtColor(src_cr, lbp_src, CV_BGR2GRAY);
			medianBlur(lbp_src, lbp_src, 3);
			lbp_src.convertTo(lbp_src, CV_64F);

			lbp::LBP lbp(8, lbp::LBP_MAPPING_NONE);
			lbp.calcLBP(lbp_src, 1);
			Mat lbpImg = lbp.getLBPImage();
			Mat mask = (lbpImg > 150);
			Mat lbpImg_m;
			lbpImg.copyTo(lbpImg_m, mask);
			cv::imshow("orig_img", lbpImg_m);
			//waitKey(0);
			Mat lbp_img;
			resize(lbpImg, lbp_img, src_small.size(), 0.0, 0.0, INTER_AREA);

			//Create feature matrix
			vector<Mat> channels;
			split(src_small, channels);
			channels.push_back(lbp_img);
			merge(channels, src_small);
			//imshow("merge", src_small);
		}

		// Computing saliency 
		BMS bms(src_small, dilation_width_1, use_normalize, handle_border, colorSpace, whitening);
		bms.computeSaliency((double)sample_step);

		Mat result = bms.getSaliencyMap();

		// Post-processing 
		if (dilation_width_2 > 0)
			dilate(result, result, Mat(), Point(-1, -1), dilation_width_2);
		if (blur_std > 0)
		{
			int blur_width = (int)MIN(floor(blur_std) * 4 + 1, 51);
			GaussianBlur(result, result, Size(blur_width, blur_width), blur_std, blur_std);
		}

		// Threshold determination
		if (loopcount == 0)
		{
			meanStdDev(result, mean, std);
			//loopcount++;
			trh = mean.at<double>(0) + stdThres * std.at<double>(0);
		}

		// Resize the saliency map
		//resize(result, result, src_cr.size());

		Mat mask_trh, masked_img;
		cv::threshold(result, masked_img, trh, 1, THRESH_BINARY);

		masked_img.convertTo(mask_trh, CV_8UC1);

		// Find contours in mask
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		cv::findContours(mask_trh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}

		//Timing
		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();

		for (int i = 0; i < boundRect.size(); i++) {
			boundRect[i].x = boundRect[i].x*(float)(src.cols / (float)(max_dimension*w / maxD));
			boundRect[i].width = boundRect[i].width*(float)(src.cols / (float)(max_dimension*w / maxD));
			boundRect[i].y = boundRect[i].y*(float)(src.rows / (float)(max_dimension*h / maxD));
			boundRect[i].height = boundRect[i].height*(float)(src.rows / (float)(max_dimension*h / maxD));
		}
		boundRectVec.push_back(boundRect);

		if (count > 10) {
			total_time += duration;
		}
		

		// Draw bonding rects 
		Mat drawing = Mat::zeros(mask_trh.size(), CV_8UC3);
		RNG rng(0xFFFFFFFF);
		Scalar color = Scalar(0, 200, 50);

		for (int i = 0; i < contours.size(); i++)
		{
			rectangle(src_cr, boundRect[i].tl(), boundRect[i].br(), color);
		}

		//Ground truth
		color = Scalar(0, 0, 200);
		//rectangle(src_cr, GT[GTcount].tl(), GT[GTcount].br(), color);

		if (count > GT_offset) {
			GTcount++;
		}
		
		//line(src_cr, HorL, HorR, Scalar(0, 0, 255),5);

		//src_cr.copyTo(src(sea_scr));

		if (saveVid) {

			video.write(src);
		}
		//std::cout << "\n Time: " << duration << std::endl;
		//Show result
		cv::imshow("Masked_img", src_cr);

		if (cv::waitKey(1) > 0)
			break;

		count++;
		//if (count == 1)
		//	break;
	}
	avg_time = total_time / (count-10);
	std::cout << "\n" << max_dimension << ", " << sample_step << ", " << stdThres << ", Average Time: " << avg_time << std::endl;
	video.release();
	destroyAllWindows();

}

void GMMDetect(cv::VideoCapture capture, vector<vector<Rect>> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio, double timeHorizon, vector<Rect> GT,int GT_offset) {
	
	double total_time = 0.0, count = 0.0, duration = 0.0;
	int GTcount = 0;

	//GPU objects
	cv::Mat src,src_small;
	capture.read(src);

	// Radar and sea windows
	cv::Rect sea_scr;
	cv::Rect radar_scr;

	RadarScreenDetect(src, radar_scr, sea_scr);

	cv::Mat radar_src = src(radar_scr);
	src = src(sea_scr);

	cv::cuda::GpuMat frame, fgmask, fgimg;
	frame.upload(src);
	fgimg.create(frame.size(), frame.type());

	cv::Ptr<cv::cuda::BackgroundSubtractorMOG> mog = cv::cuda::createBackgroundSubtractorMOG(timeHorizon, 5, backGroundRatio);
	cv::cuda::GpuMat bgimg;
	fgimg.setTo(cv::Scalar::all(0));

	//Create video writer to write file
	cv::VideoWriter video("TurnSaliency_CovTrh.avi", CV_FOURCC('M', 'J', 'P', 'G'), 15, src.size(), true);

	//Max dimension 
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w, h);
	double scale;


	while (1)
	{

		if (!capture.read(src)) {
			break;
			capture.release();
			capture = cv::VideoCapture("Videoturncropped.mp4");
			capture.read(src);
		}


		//Create LBP feature
		if (false) {
			Mat img;
			cvtColor(src, img, CV_BGR2GRAY);
			medianBlur(img, img, 11);
			img.convertTo(img, CV_64F);

			lbp::LBP lbp(4, lbp::LBP_MAPPING_NONE);
			lbp.calcLBP(img, 1);
			Mat lbpImg = lbp.getLBPImage();
			imshow("src", lbpImg);
			Mat lbp_img;
			resize(lbpImg, lbp_img, src.size(), 0.0, 0.0, INTER_AREA);
			//Create feature matrix
			vector<Mat> channels;
			split(src, channels);
			channels.push_back(lbp_img);
			merge(channels, src);

			imshow("merge", src);
			waitKey(0);
		}

		// Upload to GPU
		src = src(sea_scr);
		resize(src, src_small, Size((int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD)), 0.0, 0.0, INTER_AREA);
		scale = ((float)(max_dimension*w / maxD)* (float)(max_dimension*h / maxD)) / float((src.rows*src.cols));
		duration = static_cast<double>(cv::getTickCount());

		frame.upload(src_small);

		//cv::cuda::cvtColor(frame, frame, CV_BGR2Lab);

		std::vector<cv::cuda::GpuMat> featureMaps;

		//Whiten feature map
		cv::cuda::split(frame, featureMaps);
		for (int i = 0; i < featureMaps.size(); i++)
		{
			cv::cuda::normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, cv::NORM_MINMAX, -1);
		}
		cv::cuda::merge(featureMaps, frame);

		//Initialize result
		fgimg.setTo(cv::Scalar::all(0));
		mog->apply(frame, fgmask);
		frame.copyTo(fgimg, fgmask);

		mog->getBackgroundImage(bgimg);

		cv::Ptr<cv::cuda::Filter> filterGaus = cv::cuda::createGaussianFilter(fgmask.type(), fgmask.type(), { 3,3 }, -1);
		filterGaus->apply(fgmask, fgmask);

		//cuda::resize(fgmask, fgmask, src.size());
		//Download result from Gpu
		cv::Mat result;
		fgmask.download(result);


		// Radar Detection
		//cv::Mat radar_img = orig_img(Radar_scr);

		////std::vector<cv::Mat> channels_rad;
		//cv::split(radar_img, channels_rad);

		// 		cv::Mat radar_thr;
		//cv::threshold(channels_rad[2], radar_thr, 120, 255,CV_THRESH_BINARY);
		//imshow("Radar_trh", radar_thr);
		//waitKey(40);
		////std::cout << "ret (python)  = " << std::endl << format(radar_img, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
		//std::vector<std::vector<cv::Point> > contours_rad;
		//std::vector<cv::Vec4i> hierarchy_rad;
		//findContours(radar_thr, contours_rad, hierarchy_rad, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		////std::vector<std::vector<cv::Point> > contours_poly(contours.size());
		//std::vector<cv::Rect> boundRect_rad;// (contours_rad.size());
		//									//Radar location
		//int own_loc[2] = { 89,98 };

		//for (int i = 0; i < contours_rad.size(); i++)
		//{
		//	//approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
		//	//boundRect_rad[i] = cv::boundingRect(contours_rad[i]);
		//	boundRect_rad.push_back(cv::boundingRect(contours_rad[i]));
		//	if (i >= 2) {
		//		int tar_loc[2] = { own_loc[0] - boundRect_rad[i].x ,own_loc[1] - boundRect_rad[i].y };
		//		std::cout << ", x = " << tar_loc[0] << ", y = " << tar_loc[1] << std::endl;
		//	}
		//}


		//Contour locator
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(result, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

		double minArea = 20;
		std::vector<cv::Rect> savedBoundBox;

		std::vector<std::vector<cv::Point> > contours_poly(contours.size());// contours.size());
		std::vector<cv::Rect> boundRect(contours.size());// contours.size());

		for (int i = 0; i< contours.size(); i++)
		{
			approxPolyDP(cv::Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = cv::boundingRect(cv::Mat(contours_poly[i]));
			//int y = boundRect[i].y + boundRect[i].height / 2;

			//minArea = 200 - 0.23*y;
			double area = contourArea(contours[i]);
			if ((area > minArea))// && (frame.rows - boundRect[i].y + boundRect[i].height / 2 > 100))
			{
				savedBoundBox.push_back(boundRect[i]);
			}
		}

		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();
		

		for (int i = 0; i < savedBoundBox.size(); i++) {
			savedBoundBox[i].x = savedBoundBox[i].x*(float)(src.cols / (float)(max_dimension*w / maxD));
			savedBoundBox[i].width = savedBoundBox[i].width*(float)(src.cols / (float)(max_dimension*w / maxD));
			savedBoundBox[i].y = savedBoundBox[i].y*(float)(src.rows / (float)(max_dimension*h / maxD));
			savedBoundBox[i].height = savedBoundBox[i].height*(float)(src.rows / (float)(max_dimension*h / maxD));
		}

		boundRectVec.push_back(savedBoundBox);

		/*double maxArea = 0.0
		if ((savedBoundBox.size() == 0) && (contours.size() != 0))
		{
		for (int i = 0; i< contours.size(); i++)
		{
		double area = contourArea(contours[i]);
		if (area > maxArea)
		{
		maxArea = area;
		savedContour.push_back(i);
		}
		}
		}*/

		if (count > 10) {
			total_time += duration;
		}
	
		
		//std::cout << "\n Duration :" << duration;
		//std::cout << "  FPS: " << 1 / duration;

		// Draw bonding rects 
		cv::Mat drawing = cv::Mat::zeros(result.size(), CV_8UC3);

		cv::Scalar color = cv::Scalar(0, 200, 50);

		for (int i = 0; i < savedBoundBox.size(); i++)
		{
			cv::rectangle(src, savedBoundBox[i].tl(), savedBoundBox[i].br(), color);
		}
		//resize(src_small, src_small, src.size());

		

		//Ground truth
		color = Scalar(0, 0, 200);
		//cv::rectangle(src, GT[GTcount].tl(), GT[GTcount].br(), color);
		if (count > GT_offset) {
			GTcount++;
		}
		//cv::waitKey(30);
		cv::imshow("video", src);
		//cv::imshow("res", result);
		

		if (false) {

			video.write(src);
		}

		count++;
		//cv::waitKey(50);
		if (cv::waitKey(1) > 0)
			break;
	
		
		//if (count == 1)
		//	break;
	}


	//duration1 = static_cast<double>(cv::getTickCount()) - duration1;
	//duration1 /= cv::getTickFrequency();
	avg_time = total_time / (count-10);

	std::cout << "\n" << max_dimension << ", " << backGroundRatio << ", "<< timeHorizon << ", Average Time: " << avg_time << std::endl;

	destroyAllWindows();
	video.release();
}