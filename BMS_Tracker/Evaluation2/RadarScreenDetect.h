#include "stdafx.h"
#include <iostream>
#include <set>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <ctime>
#include <string>
#include <vector>
#include <omp.h>

cv::Point RadarScreenDetect(cv::Mat src, cv::Rect& radar_scr, cv::Rect& sea_scr, int& radius);
