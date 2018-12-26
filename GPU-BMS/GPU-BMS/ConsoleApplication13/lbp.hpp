#ifndef LBP_HPP_
#define LBP_HPP_

//! \author philipp <bytefish[at]gmx[dot]de>
//! \copyright BSD, see LICENSE.

#include "opencv2/opencv.hpp"
#include <limits>

using namespace cv;
using namespace std;

namespace lbp {

// templated functions
template <typename _Tp>
void OLBP_(const cv::Mat& src, cv::Mat& dst);

template <typename _Tp>
void ELBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

template <typename _Tp>
void VARLBP_(const cv::Mat& src, cv::Mat& dst, int radius = 1, int neighbors = 8);

template <typename _Tp>
void variance_lbp_(const cv::Mat& src,cv::Mat& _m2, int radius = 1, int neighbors = 8);

// wrapper functions
void OLBP(const Mat& src, Mat& dst);
void ELBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);
void VARLBP(const Mat& src, Mat& dst, int radius = 1, int neighbors = 8);
void variance_lbp(const Mat& src, Mat& _m2 ,int radius = 1, int neighbors = 8);

// Mat return type functions
Mat OLBP(const Mat& src);
Mat ELBP(const Mat& src, int radius = 1, int neighbors = 8);
Mat VARLBP(const Mat& src, int radius = 1, int neighbors = 8);
Mat variance_lbp(const Mat& src, int radius = 1, int neighbors = 8);
}
#endif
