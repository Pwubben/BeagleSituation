
#ifndef BMS_H
#define BMS_H

#ifdef IMDEBUG
#include <imdebug.h>
#endif
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

static const int CL_RGB = 1;
static const int CL_Lab = 2;
static const int CL_Luv = 4;

static cv::RNG BMS_RNG;

class BMS
{
public:
	BMS(const cv::cuda::GpuMat& src, int dw1, bool nm, bool hb, int colorSpace, bool whitening);

	cv::cuda::GpuMat getSaliencyMap();
	void computeSaliency(double step);
private:
	cv::cuda::GpuMat mSaliencyMap;
	int mAttMapCount;
	cv::cuda::GpuMat mBorderPriorMap;
	cv::cuda::GpuMat mSrc, am, bm, ret, map1, map2, mapres;
	std::vector<cv::cuda::GpuMat> mFeatureMaps;
	int mDilationWidth_1;
	bool mHandleBorder;
	bool mNormalize;
	bool mWhitening;
	int mColorSpace;
	cv::cuda::GpuMat getAttentionMap(const cv::cuda::GpuMat& bm, int dilation_width_1, bool toNormalize, bool handle_border);
	void whitenFeatMap(const cv::cuda::GpuMat& img, float reg);
	//void computeBorderPriorMap(float reg, float marginRatio);
};

//void postProcessByRec8u(cv::Mat& salmap, int kernelWidth);
//void postProcessByRec(cv::Mat& salmap, int kernelWidth);



#endif



