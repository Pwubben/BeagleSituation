void SaliencyDetect(cv::VideoCapture capture, vector<Rect> &boundRect, double &avg_time, float max_dimension, int sample_step, double stdThres);

void GMMDetect(cv::VideoCapture capture, vector<Rect> &boundRect, double &avg_time, float max_dimension, double stdThres);
