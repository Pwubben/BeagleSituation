#include "stdafx.h"
#include "BMS.h"
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>

using namespace cv;
using namespace std;

#define COV_MAT_REG 50.0f
clock_t tt;
BMS::BMS(const Mat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening)
	:mDilationWidth_1(dw1), mNormalize(nm), mHandleBorder(hb), mAttMapCount(0), mColorSpace(colorSpace), mWhitening(whitening)
{
	mSrc = src.clone();
	mSaliencyMap = Mat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);

	if (CL_RGB & colorSpace)
		whitenFeatMap(mSrc, COV_MAT_REG);
	if (CL_Lab & colorSpace)
	{
		Mat lab;
		cvtColor(mSrc, lab, CV_RGB2Lab);
		whitenFeatMap(lab, COV_MAT_REG);
		
	}
	if (CL_Luv & colorSpace)
	{
		Mat luv;
		cvtColor(mSrc, luv, CV_RGB2Luv);
		whitenFeatMap(luv, COV_MAT_REG);
	}
}

void BMS::computeSaliency(double step)
{
  //#pragma omp parallel for //collapse(2)
	/*for (int i = 0; i<mFeatureMaps.size(); ++i)
	{*/
		
		/*double max_, min_;
		minMaxLoc(mFeatureMaps[0], &min_, &max_);
		int max = floor(max_);
		int min = floor(min_);*/
		//cv::imshow("src", mSrc);
		//waitKey(0);
		//imwrite("../../images/Gray_Image.jpg", mSrc);

		for (int thresh = 255; 0 < thresh; thresh -= 20) {
			Mat bm = (mFeatureMaps[1] > thresh - step) & (mFeatureMaps[1]<thresh);
			Mat maskImg;
			Mat prjImg = mSrc.clone();
			prjImg.copyTo(maskImg, bm);
			imshow("res",maskImg);
			waitKey(0);
			destroyAllWindows;
		}
	//	for (int thresh = min; thresh < max; thresh += step)
	//	{

	//		Mat bm = (mFeatureMaps[i] > thresh - step) & (mFeatureMaps[i]<thresh);
	//		//bm = mFeatureMaps[i] < thresh;
	//		//imshow("Boolean Map", bm);
	//		//waitKey(30);
	//		Mat am = getAttentionMap(bm, mDilationWidth_1, mNormalize, mHandleBorder);

	//	
	//		//imshow("Attention Map", am);
	//		//cv::waitKey(0);
	//		mSaliencyMap += am;
	//		//imshow("Saliency Map", mSaliencyMap);

	//		mAttMapCount++;
	//		//waitKey(30);
	//	}
	//}
}


cv::Mat BMS::getAttentionMap(const cv::Mat& bm, int dilation_width_1, bool toNormalize, bool handle_border)
{
	Mat ret = bm.clone();
	/*int jump;
	if (handle_border)
	{
		for (int i=0;i<bm.rows;i++)
		{
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(i,0+jump)!=1)
				floodFill(ret,Point(0+jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
			jump = BMS_RNG.uniform(0.0,1.0)>0.99 ?BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(i,bm.cols-1-jump)!=1)
				floodFill(ret,Point(bm.cols-1-jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
		for (int j=0;j<bm.cols;j++)
		{
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(0+jump,j)!=1)
				floodFill(ret,Point(j,0+jump),Scalar(1),0,Scalar(0),Scalar(0),8);
			jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
			if (ret.at<uchar>(bm.rows-1-jump,j)!=1)
				floodFill(ret,Point(j,bm.rows-1-jump),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
	}
	else
	{
		#pragma omp parallel for 
		for (int i=0;i<bm.rows;i++)
		{
			if (ret.at<uchar>(i,0)!=1)
				floodFill(ret,Point(0,i),Scalar(1),0,Scalar(0),Scalar(0),8);
			if (ret.at<uchar>(i,bm.cols-1)!=1)
				floodFill(ret,Point(bm.cols-1,i),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
		#pragma omp parallel for 
		for (int j=0;j<bm.cols;j++)
		{
			if (ret.at<uchar>(0,j)!=1)
				floodFill(ret,Point(j,0),Scalar(1),0,Scalar(0),Scalar(0),8);
			if (ret.at<uchar>(bm.rows-1,j)!=1)
				floodFill(ret,Point(j,bm.rows-1),Scalar(1),0,Scalar(0),Scalar(0),8);
		}
	}*/

	//cout << "ret (python)  = " << endl << format(ret, Formatter::FMT_PYTHON) << endl << endl;
	ret = ret != 1;

	Mat map1 = ret & bm;
	Mat map2 = ret & (~bm);
 
	
	if (dilation_width_1 > 0)
	{
		dilate(map1, map1, Mat(), Point(-1, -1), dilation_width_1);
		dilate(map2, map2, Mat(), Point(-1, -1), dilation_width_1);
	}

	map1.convertTo(map1, CV_32FC1);
	map2.convertTo(map2, CV_32FC1);    

	if (toNormalize)
	{
		normalize(map1, map1, 1.0, 0.0, NORM_L2);
		normalize(map2, map2, 1.0, 0.0, NORM_L2);
	}
	else
		normalize(ret, ret, 0.0, 1.0, NORM_MINMAX);

	return map1 + map2;

}

Mat BMS::getSaliencyMap()
{
	//cout << "ret (python)  = " << endl << format(mSaliencyMap, Formatter::FMT_PYTHON) << endl << endl;

	Mat ret = Mat::zeros(mSrc.size(), CV_32FC1);

	double min, max,cl;

	minMaxLoc(mSaliencyMap, &min, &max);

	cout << "\n"<< min << ", " << max << endl;
	
	//normalize(mSaliencyMap, ret, 255.0, 0.0, NORM_MINMAX);
	ret = mSaliencyMap;
	//ret.convertTo(ret, CV_8UC1);
	return ret;
}

void BMS::whitenFeatMap(const cv::Mat& img, float reg)
{
	assert(img.channels() == 3 && img.type() == CV_8UC3);

	vector<Mat> featureMaps;

	if (!mWhitening)
	{
		split(img, featureMaps);
		for (int i = 0; i < featureMaps.size(); i++)
		{
			normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
			medianBlur(featureMaps[i], featureMaps[i], 3);
			mFeatureMaps.push_back(featureMaps[i]);
		}
		return;
	}

	Mat srcF, meanF, covF;
	img.convertTo(srcF, CV_32FC3);
	Mat samples = srcF.reshape(1, img.rows*img.cols);
	calcCovarMatrix(samples, covF, meanF, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE, CV_32F);

	covF += Mat::eye(covF.rows, covF.cols, CV_32FC1)*reg;
	SVD svd(covF);
	Mat sqrtW;
	sqrt(svd.w, sqrtW);
	Mat sqrtInvCovF = svd.u * Mat::diag(1.0 / sqrtW);

	Mat whitenedSrc = srcF.reshape(1, img.rows*img.cols)*sqrtInvCovF;
	whitenedSrc = whitenedSrc.reshape(3, img.rows);

	split(whitenedSrc, featureMaps);

	for (int i = 0; i < featureMaps.size(); i++)
	{
		normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX);
		featureMaps[i].convertTo(featureMaps[i], CV_8U);
		medianBlur(featureMaps[i], featureMaps[i], 3);
		mFeatureMaps.push_back(featureMaps[i]);
	}
}