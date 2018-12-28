void SaliencyDetect(cv::VideoCapture capture, std::vector<std::vector<cv::Rect>> &boundRectVec, double &avg_time, double max_dimension, double sample_step, double stdThres);

void GMMDetect(cv::VideoCapture capture, std::vector<cv::Rect> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio);

//Ground Truth Generation
void GroundTruth(cv::VideoCapture capture, std::vector<cv::Rect> &boundRectVec);

//Evaluation
void readGroundTruth(std::string file, std::vector<std::vector<int>> &groundTruth);

void readBoundRectData(std::string fileName, std::vector<std::vector<std::vector<std::vector<int>>>>& boundRectvec);

bool doOverlap(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);