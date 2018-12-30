void SaliencyDetect(cv::VideoCapture capture, std::vector<std::vector<cv::Rect>> &boundRectVec, double &avg_time, double max_dimension, double sample_step, double stdThres);

void GMMDetect(cv::VideoCapture capture, std::vector<std::vector<cv::Rect>> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio);

//Ground Truth Generation
void GroundTruth(cv::VideoCapture capture, std::vector<cv::Rect> &boundRectVec);

//Evaluation
void readGroundTruth(std::string file, std::vector<std::vector<int>> &groundTruth);

void readBoundRectData(std::string fileName, std::vector<std::vector<std::vector<std::vector<int>>>>& boundRectvec);

bool doOverlap(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);

void trueFalsePositiveRate(std::vector<std::vector<std::vector<std::vector<int>>>> boundRectData, std::vector<std::vector<int>> groundTruth, std::vector<std::vector<int>> &falsePositiveCount, std::vector<std::vector<int>> &truePositiveCount, std::vector<std::vector<double>> &precision, std::vector<double> &recall, std::vector<std::vector<double>> &AoU);

double IntersectionOverUnion(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);

void writeBoundRectFile(std::vector<std::vector<std::vector<cv::Rect>>> boundRectData, std::ofstream &File);

std::string getFileString(std::string fileName);
	