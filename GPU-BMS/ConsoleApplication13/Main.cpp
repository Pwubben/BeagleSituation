// ConsoleApplication13.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <conio.h>
#include "lbp.hpp"
#include "opencv2/opencv.hpp"
#include "BMS.h"

//#define MAX_IMG_DIM 600

using namespace cv;
using namespace std;

int main(int args, char** argv)
{
	int sample_step = 25;
	int dilation_width_1 = 7;
	int dilation_width_2 = 7;
	float blur_std = 9;
	bool use_normalize = 1;
	bool handle_border = 0;
	int colorSpace = 2;
	bool whitening = 0;
	float max_dimension = 800;
	double duration,total_time = 0.0, count = 0.0;
	clock_t ttt;
	double avg_time = 0;

	int i, j, k;
	i = j = k = 0;

	// Declare VideoCapture object for storing video
	cv::Mat src;
	cv::VideoCapture capture("Videoturncropped.mp4");
	capture.read(src);


	// Preprocessing
	//Mat src_small;
	float w = (float)src.cols, h = (float)src.rows;
	float maxD = max(w, h);
	cv::cuda::setDevice(0);
	Size size((max_dimension*w / maxD), (max_dimension*h / maxD));
	while (1)
	{
		duration = static_cast<double>(cv::getTickCount());

		capture >> src;

		if (src.empty())
		{
			// Reach end of the video file
			break;
		}
		Mat res_cpu;
		
		// Upload to GPU
		cv::cuda::GpuMat src_gpu, src_small;
		

		src_small.upload(src);
		
		//std::cout << "  FPS: " << 1 / duration;
		// Resize image
		//cv::cuda::resize(src_gpu, src_small, size, 0.0, 0.0, INTER_AREA);
		
		//resize(src, src_small, Size((int)(max_dimension*w / maxD), (int)(max_dimension*h / maxD)), 0.0, 0.0, INTER_AREA);

		// LBP computation
		
		//int radius = 3;
		//int neighbors = 8;
		//cv::cuda::GpuMat lbp_feature,lbp_src;
		//cuda::cvtColor(src_small, lbp_src, CV_BGR2GRAY);
		//lbp::VARLBP(lbp_src, lbp_feature);// , radius, neighbors);
		//normalize(lbp_feature, lbp_feature, 0, 255, NORM_MINMAX, CV_8UC1);
		//resize(lbp_feature, lbp_feature, src.size());
		
		// Computing saliency 

		BMS bms(src_small, dilation_width_1, use_normalize, handle_border, colorSpace, whitening);


		bms.computeSaliency((double)sample_step);

		cuda::GpuMat result = bms.getSaliencyMap();


		/*Mat featurevec;
		featurevec = result + lbp_feature;
		normalize(featurevec,featurevec, 0, 255, NORM_MINMAX, CV_8UC1);*/
		//
		//imshow("Res", result);
		//imshow("Res2", lbp_feature);
		//vector<Mat> channels;
		//
		//split(result, channels);
		//
		//channels.push_back(lbp_feature);
		//merge(channels, featurevec);
		//
	/*	imshow("Merge1", lbp_feature);
		imshow("Merge2", result);
		imshow("Merge", featurevec);
		waitKey(0);*/
 
		// Post-processing 
		
		
		result.download(res_cpu);
		duration = static_cast<double>(cv::getTickCount()) - duration;
		duration /= cv::getTickFrequency();
		total_time += duration;
		count++;

		std::cout << "time: " << duration << endl;

		if (dilation_width_2 > 0)
			dilate(res_cpu, res_cpu, Mat(), Point(-1, -1), dilation_width_2);
		if (blur_std > 0)
		{
			int blur_width = (int)MIN(floor(blur_std) * 4 + 1, 51);
			GaussianBlur(res_cpu, res_cpu, Size(blur_width, blur_width), blur_std, blur_std);
		}

		// Resize the saliency map
		//resize(res_cpu, res_cpu, src.size());

		Mat mask_trh;
		threshold(res_cpu, mask_trh, 180, 255, THRESH_BINARY);
		Mat masked_img;
		src.copyTo(masked_img, mask_trh);

		
		//imshow("src_small", src_small);
		//imshow("Boolean map", result);
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(mask_trh, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		vector<vector<Point> > contours_poly(contours.size());
		vector<Rect> boundRect(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}
		/// Draw polygonal contour + bonding rects + circles
		Mat drawing = Mat::zeros(mask_trh.size(), CV_8UC3);
		RNG rng(0xFFFFFFFF);
		Scalar color = Scalar(0, 200, 50);

		for (int i = 0; i < contours.size(); i++)
		{
			rectangle(src, boundRect[i].tl(), boundRect[i].br(), color);
		}

		//Timing




		imshow("Masked_img", src);

		if (cv::waitKey(30) > 0)
			break;
	}


	return 0;
}