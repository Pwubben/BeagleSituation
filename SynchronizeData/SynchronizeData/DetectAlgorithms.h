using namespace std;
using namespace cv;

void SaliencyDetect(cv::VideoCapture capture, vector<vector<Rect>> &boundRectVec, double &avg_time, double max_dimension, double sample_step, double stdThres);

void GMMDetect(cv::VideoCapture capture, vector<Rect> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio);

void GroundTruth(cv::VideoCapture capture, vector<Rect> &boundRectVec);
