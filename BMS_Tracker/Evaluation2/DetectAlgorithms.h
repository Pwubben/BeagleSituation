using namespace std;
using namespace cv;


void SaliencyDetect(cv::VideoCapture capture, double &avg_time, double max_dimension, double sample_step, double stdThres,std::vector<cv::Rect> GT,int GT_offset, int stopFrame);

vector<Point2f> RadarDetection(Mat radar_img, cv::Point center, int radius);

void GMMDetect(cv::VideoCapture capture, std::vector<std::vector<cv::Rect>> &boundRectVec, double &avg_time, float max_dimension, double backGroundRatio, double timeHorizon, std::vector<cv::Rect> GT,int GT_offset,vector<vector<double>> &falsePositiveCountGMM, vector<vector<double>> &truePositiveCountGMM, vector<vector<double>> & trueNegativeCountGMM, vector<vector<double>> & falseNegativeCountGMM, vector<vector<double>> &falsePositiveAreaGMM, vector<vector<double>> &truePositiveAreaGMM, vector<vector<double>> &precisionAreaGMM, vector<vector<double>> &IoUGMM,int stopFrame);

void DataGeneration(std::string videoFile, std::string groundTruthFile, std::string avgTimeFile, std::string ResultFile, int GT_offset, int stopFrame);

void writeResultFile(std::vector<std::vector<int>> falsePositiveCount, std::vector<std::vector<int>> truePositiveCount, std::vector<std::vector<double>> precision, std::vector<double> recall, std::vector<std::vector<double>> IoU, std::ofstream &File);

//Evaluation
void readGroundTruth(std::string file, std::vector<std::vector<int>> &groundTruth);

std::string getFileString(std::string fileName);

void writeFileNames(std::string File, ::string& videoFile, std::string& boundRectFileSal, std::string& avgTimeFileSal, std::string& boundRectFileGMM, std::string& avgTimeFileGMM, std::string& labelFile);

//Storage
void HorizonDetect(Mat src);
	