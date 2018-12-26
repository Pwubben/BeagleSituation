#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include "lbp.hpp"
#include "opencv2/opencv.hpp"
#include "trim.h"
#include "BMS.h"
#include "RadarScreenDetect.h"

using namespace cv;
using namespace std;

void SaliencyDetect(cv::VideoCapture capture, vector<Rect> &boundRect,double &avg_time, float max_dimension,int sample_step,double stdThres)
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
	double total_time = 0.0, avg_time, count = 0.0, duration;
	int  loopcount = 0;

	Mat left_vec, right_vec;
	Point maxL, maxR, minL, minR;
	double maxValL, minValL, maxValR, minValR;

	// Thresholding variables
	double trh = 180;
	Mat mean, std;
	Mat src_gray;


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
		resize(result, result, src_cr.size());

		Mat mask_trh, masked_img;
		cv::threshold(result, masked_img, trh, 1, THRESH_BINARY);

		masked_img.convertTo(mask_trh, CV_8UC1);

		// Find contours in mask
		vector<vector<Point> > contours;
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
		total_time += duration;
		count++;

		// Draw bonding rects 
		Mat drawing = Mat::zeros(mask_trh.size(), CV_8UC3);
		RNG rng(0xFFFFFFFF);
		Scalar color = Scalar(0, 200, 50);

		for (int i = 0; i < contours.size(); i++)
		{
			rectangle(src_cr, boundRect[i].tl(), boundRect[i].br(), color);
		}
		//line(src_cr, HorL, HorR, Scalar(0, 0, 255),5);

		//src_cr.copyTo(src(sea_scr));

		if (saveVid) {

			video.write(src);
		}

		//Show result
		cv::imshow("Masked_img", src_cr);

		if (cv::waitKey(1) > 0)
			break;
	}
	avg_time = total_time / count;
	std::cout << "\n Average Time: " << avg_time << std::endl;
	video.release();
	getch();
	return 0;
}