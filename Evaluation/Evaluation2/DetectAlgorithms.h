using namespace std;
using namespace cv;

void SaliencyDetect(cv::VideoCapture capture, std::vector<std::vector<cv::Rect>> &boundRectVec, double &avg_time, double max_dimension, double sample_step, double stdThres,std::vector<cv::Rect> GT,int GT_offset, vector<vector<double>>& falsePositiveCountSal, vector<vector<double>>& truePositiveCountSal, vector<vector<double>>& precisionCountSal, vector<vector<double>>& falsePositiveAreaSal, vector<vector<double>>& truePositiveAreaSal, vector<vector<double>>& precisionAreaSal, vector<vector<double>>& IoUSal);

void GMMDetect(cv::VideoCapture capture, std::vector<std::vector<cv::Rect>> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio, double timeHorizon, std::vector<cv::Rect> GT,int GT_offset,vector<vector<double>> &falsePositiveCountGMM, vector<vector<double>> &truePositiveCountGMM, vector<vector<double>> & trueNegativeCountGMM, vector<vector<double>> & falseNegativeCountGMM, vector<vector<double>> &falsePositiveAreaGMM, vector<vector<double>> &truePositiveAreaGMM, vector<vector<double>> &precisionAreaGMM, vector<vector<double>> &IoUGMM,int stopFrame);

void DataGeneration(std::string videoFile, std::string groundTruthFile, std::string boundRectFileSal, std::string avgTimeFileSal, std::string boundRectFileGMM, std::string avgTimeFileGMM, std::string labelFile, int GT_offset, int stopFrame);

void writeFileNames(std::string File, std::string& videoFile, std::string& boundRectFileSal, std::string& avgTimeFileSal, std::string& boundRectFileGMM, std::string& avgTimeFileGMM, std::string& labelFile);

void writeResultFile(std::vector<std::vector<int>> falsePositiveCount, std::vector<std::vector<int>> truePositiveCount, std::vector<std::vector<double>> precision, std::vector<double> recall, std::vector<std::vector<double>> IoU, std::ofstream &File);

void trueFalsePositiveRateROC(vector<Rect> boundRectData, Rect groundTruth, vector<vector<double>> &falsePositiveCount, vector<vector<double>> &truePositiveCount,
	vector<vector<double>> &precisionCount, vector<vector<double>> &falsePositiveArea, vector<vector<double>> &truePositiveArea,
	vector<vector<double>> &precisionArea, vector<vector<double>> &IoU, int GT_offset, int timeCount);

void trueFalsePositiveRateROCGMM(vector<Rect> boundRectData, Rect groundTruth, vector<vector<double>> &falsePositiveCount, vector<vector<double>> &truePositiveCount, vector<vector<double>> &trueNegativeCount,
	vector<vector<double>> &falseNegativeCount, vector<vector<double>> &falsePositiveArea, vector<vector<double>> &truePositiveArea,
	vector<vector<double>> &precisionArea, vector<vector<double>> &IoU, int GT_offset, int timeCount);

//Ground Truth Generation
void GroundTruth(cv::VideoCapture capture, std::vector<cv::Rect> &boundRectVec);

//Evaluation
void readGroundTruth(std::string file, std::vector<std::vector<int>> &groundTruth);

void readBoundRectData(std::string fileName, std::vector<std::vector<std::vector<std::vector<int>>>>& boundRectvec);

bool doOverlap(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);

void trueFalsePositiveRate(std::vector<std::vector<std::vector<std::vector<int>>>> boundRectData, std::vector<std::vector<int>> groundTruth, std::vector<std::vector<int>> &falsePositiveCount, std::vector<std::vector<int>> &truePositiveCount, std::vector<std::vector<double>> &precision, std::vector<double> &recall, std::vector<std::vector<double>> &AoU, int GT_offset);

double IntersectionOverUnion(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);

void writeBoundRectFile(std::vector<std::vector<std::vector<cv::Rect>>> boundRectData, std::ofstream &File);

std::string getFileString(std::string fileName);
	
double IntersectionArea(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);

double nonIntersect(cv::Point l1, cv::Point r1, cv::Point l2, cv::Point r2);