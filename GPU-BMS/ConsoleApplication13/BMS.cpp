#include "stdafx.h"
#include "BMS.h"
#include <vector>
#include <cmath>
#include <ctime>
#include <omp.h>
using namespace cv;
using namespace std;

double duration, total_time = 0.0, count = 0.0;

#define COV_MAT_REG 50.0f
clock_t tt;
BMS::BMS(const cuda::GpuMat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening)
	:mDilationWidth_1(dw1), mNormalize(nm), mHandleBorder(hb), mAttMapCount(0), mColorSpace(colorSpace), mWhitening(whitening)
{
	mSrc = src.clone();
	/*mSaliencyMap = cuda::GpuMat::zeros(src.size(), CV_32FC1);
	mBorderPriorMap = Mat::zeros(src.size(), CV_32FC1);*/
	mSaliencyMap = cuda::GpuMat(mSrc.size(), CV_32FC1);
	mSaliencyMap.setTo(Scalar::all(0));
	bm = cuda::GpuMat(mSrc.size(), CV_32FC1);
	am = cuda::GpuMat(mSrc.size(), CV_32FC1);
	ret = cuda::GpuMat(mSrc.size(), CV_32FC1);
	map1 = cuda::GpuMat(mSrc.size(), CV_32FC1); 
	map2 = cuda::GpuMat(mSrc.size(), CV_32FC1); 
	mapres =cuda::GpuMat(mSrc.size(), CV_32FC1);
	//cuda::GpuMat mBorderPriorMap;


	if (CL_RGB & colorSpace)
		whitenFeatMap(mSrc, COV_MAT_REG);
	if (CL_Lab & colorSpace)
	{
		cuda::GpuMat lab(mSrc.size(), CV_32FC1);
		cuda::cvtColor(mSrc, lab, CV_RGB2Lab);
		whitenFeatMap(lab, COV_MAT_REG);
	}
	if (CL_Luv & colorSpace)
	{
		cuda::GpuMat luv;
		cuda::cvtColor(mSrc, luv, CV_RGB2Luv);
		whitenFeatMap(luv, COV_MAT_REG);
	}
}

void BMS::computeSaliency(double step)
{
  //#pragma omp parallel for 
	for (int i = 0; i<mFeatureMaps.size(); ++i)
	{
		
		for (int thresh = 0; thresh < 255; thresh += step)
		{
			cuda::threshold(mFeatureMaps[i], bm, thresh, 255, THRESH_BINARY);
			//bm = (mFeatureMaps[i] > thresh - step) & (mFeatureMaps[i]<thresh);

			am = getAttentionMap(bm, mDilationWidth_1, mNormalize, mHandleBorder);

			//imshow("Attention Map", am);
			cuda::add(mSaliencyMap,am,mSaliencyMap);
			Mat comp;
			mSaliencyMap.download(comp);
			imshow("Saliency Map", comp);


		}
	}
}


cv::cuda::GpuMat BMS::getAttentionMap(const cv::cuda::GpuMat& bm, int dilation_width_1, bool toNormalize, bool handle_border)
{
	ret = bm.clone();
	//int jump;
	//if (handle_border)
	//{
	//	for (int i=0;i<bm.rows;i++)
	//	{
	//		jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
	//		if (ret.at<uchar>(i,0+jump)!=1)
	//			floodFill(ret,Point(0+jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
	//		jump = BMS_RNG.uniform(0.0,1.0)>0.99 ?BMS_RNG.uniform(5,25):0;
	//		if (ret.at<uchar>(i,bm.cols-1-jump)!=1)
	//			floodFill(ret,Point(bm.cols-1-jump,i),Scalar(1),0,Scalar(0),Scalar(0),8);
	//	}
	//	for (int j=0;j<bm.cols;j++)
	//	{
	//		jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
	//		if (ret.at<uchar>(0+jump,j)!=1)
	//			floodFill(ret,Point(j,0+jump),Scalar(1),0,Scalar(0),Scalar(0),8);
	//		jump= BMS_RNG.uniform(0.0,1.0)>0.99 ? BMS_RNG.uniform(5,25):0;
	//		if (ret.at<uchar>(bm.rows-1-jump,j)!=1)
	//			floodFill(ret,Point(j,bm.rows-1-jump),Scalar(1),0,Scalar(0),Scalar(0),8);
	//	}
	//}
	//else
	//{
	//	for (int i=0;i<bm.rows;i++)
	//	{
	//		if (ret.at<uchar>(i,0)!=1)
	//			floodFill(ret,Point(0,i),Scalar(1),0,Scalar(0),Scalar(0),8);
	//		if (ret.at<uchar>(i,bm.cols-1)!=1)
	//			floodFill(ret,Point(bm.cols-1,i),Scalar(1),0,Scalar(0),Scalar(0),8);
	//	}
	//	for (int j=0;j<bm.cols;j++)
	//	{
	//		if (ret.at<uchar>(0,j)!=1)
	//			floodFill(ret,Point(j,0),Scalar(1),0,Scalar(0),Scalar(0),8);
	//		if (ret.at<uchar>(bm.rows-1,j)!=1)
	//			floodFill(ret,Point(j,bm.rows-1),Scalar(1),0,Scalar(0),Scalar(0),8);
	//	}
	//}

	//cout << "ret (python)  = " << endl << format(ret, Formatter::FMT_PYTHON) << endl << endl;
	//ret = ret != 1;


	cuda::compare(ret, 1, ret, CMP_NE);

	//map1(mSrc.size(), CV_32FC1);
	//map2(mSrc.size(), CV_32FC1);
	cuda::compare(ret, bm, map1, CMP_EQ);
	cuda::compare(ret, bm, map2, CMP_NE);
	//map1 = ret & bm;
	//map2 = ret & (~bm);
	//Mat element = cv::getStructuringElement(MORPH_RECT, Size(2 * dilation_width_1 + 1, 2 * dilation_width_1 + 1));
	//cuda::GpuMat d_element(element);
	//Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, map1.type(), element);
	//
	//if (dilation_width_1 > 0)
	//{
	//	dilateFilter->apply(map1, map1);
	//	dilateFilter->apply(map2, map2);
	//	/*cuda::dilate(map1, map1, Mat(), Point(-1, -1), dilation_width_1);
	//	dilate(map2, map2, Mat(), Point(-1, -1), dilation_width_1);*/
	//}

	map1.convertTo(map1, CV_32FC1);
	map2.convertTo(map2, CV_32FC1);

	if (toNormalize)
	{
		cuda::normalize(map1, map1, 1.0, 0.0, NORM_L2,-1);
		cuda::normalize(map2, map2, 1.0, 0.0, NORM_L2,-1);
	}
	else
		cuda::normalize(ret, ret, 0.0, 1.0, NORM_MINMAX,-1);

	//mapres(mSrc.size(), CV_32FC1);
	cuda::add(map1, map2, mapres);
	return mapres;
}

cuda::GpuMat BMS::getSaliencyMap()
{
	cuda::GpuMat ret(mSrc.size(), CV_32FC1);
	cuda::normalize(mSaliencyMap, ret, 255.0, 0.0, NORM_MINMAX,-1);
	ret.convertTo(ret, CV_8UC1);
	return ret;
}

void BMS::whitenFeatMap(const cv::cuda::GpuMat& img, float reg)
{
	//assert(img.channels() == 3 && img.type() == CV_8UC3);

	vector<cuda::GpuMat> featureMaps;
	
	if (!mWhitening)
	{
		cuda::split(img, featureMaps);
		for (int i = 0; i < featureMaps.size(); i++)
		{
			cuda::normalize(featureMaps[i], featureMaps[i], 255.0, 0.0, NORM_MINMAX,-1);

			//cuda::createBoxFilter(featureMaps[i], featureMaps[i], 3);//  medianBlur(featureMaps[i], featureMaps[i], 3);
			mFeatureMaps.push_back(featureMaps[i]);
		}
		return;
	}
/*
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
		featureMaps[i].convertTo(featureMaps[i], CV_8UC3);
		medianBlur(featureMaps[i], featureMaps[i], 3);
		mFeatureMaps.push_back(featureMaps[i]);
	}*/
}